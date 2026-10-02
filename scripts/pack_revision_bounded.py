"""Stream an archive to a capped file while retaining a disk reserve.

Unlike the conservative canonical packer, this does not require disk space for
an uncompressed copy of all inputs. A reader thread enforces the compressed byte
limit and checks free space before every write; failed partials are retained.
"""
import hashlib
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path

from package_revision_task import verify_members
from revision_package_assets import verified_asset_paths
from wasman.learning.revision_protocol import sha256


def verify_parts(paths, expected):
    producer = subprocess.Popen(['cat', '--', *map(str, paths)], stdout=subprocess.PIPE)
    decoder = subprocess.Popen(['zstd', '-q', '-d', '-c'], stdin=producer.stdout, stdout=subprocess.PIPE)
    producer.stdout.close()
    found = {}
    try:
        with tarfile.open(fileobj=decoder.stdout, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in found:
                    raise ValueError('Unexpected archive member')
                found[member.name] = hashlib.file_digest(archive.extractfile(member), 'sha256').hexdigest()
        while decoder.stdout.read(1024**2):
            pass
        if decoder.wait() or producer.wait() or found != expected:
            raise ValueError('Multipart archive member mismatch')
    finally:
        decoder.stdout.close()
        for process in (decoder, producer):
            if process.poll() is None:
                process.kill()
            process.wait()


def pack(target, files, *, reserve_bytes=40 * 2**30, max_output_bytes=None, part_bytes=None):
    target = Path(target)
    temporary = target.with_suffix(target.suffix + '.partial')
    if target.exists() or temporary.exists():
        raise ValueError('Preserve existing archive or partial')
    if part_bytes is not None:
        if part_bytes <= 0:
            raise ValueError('Positive part size required')
        if list(target.parent.glob(target.name + '.part-*')):
            raise ValueError('Preserve existing transport parts')
    available = shutil.disk_usage(target.parent).free - reserve_bytes
    # Reserve additional headroom for concurrent small restoration scratch files.
    limit = available - 2 * 2**30
    if max_output_bytes is not None:
        limit = min(limit, max_output_bytes)
    if limit <= 0:
        raise RuntimeError('Insufficient space above the disk reserve')
    expected = {name: sha256(path) for name, path in files.items()}
    for name, path in files.items():
        rel = Path(name)
        if rel.is_absolute() or '..' in rel.parts or path.is_symlink() or not path.is_file():
            raise ValueError('Unsafe archive member')
    process = subprocess.Popen(['zstd', '-q', '-T2', '-3', '-c'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    errors = []
    parts = []
    whole_hash = hashlib.sha256()

    def consume():
        written = 0
        output = None
        part_written = 0
        try:
            if part_bytes is None:
                output = temporary.open('xb')
            while block := process.stdout.read(1024**2):
                if written + len(block) > limit or shutil.disk_usage(target.parent).free < reserve_bytes + len(block):
                    raise RuntimeError('Compressed output would exceed byte limit or disk reserve')
                whole_hash.update(block)
                written += len(block)
                while block:
                    if output is None:
                        path = target.with_name(f'{target.name}.part-{len(parts):04d}.partial')
                        output = path.open('xb')
                        parts.append(path)
                        part_written = 0
                    count = len(block) if part_bytes is None else min(len(block), part_bytes - part_written)
                    output.write(block[:count])
                    block = block[count:]
                    part_written += count
                    if part_bytes is not None and part_written == part_bytes:
                        output.close()
                        output = None
        except BaseException as exc:
            errors.append(exc)
            process.kill()
        finally:
            if output is not None:
                output.close()

    consumer = threading.Thread(target=consume)
    consumer.start()
    producer_error = None
    try:
        with tarfile.open(fileobj=process.stdin, mode='w|') as archive:
            for name, path in sorted(files.items()):
                info = archive.gettarinfo(str(path), arcname=name)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                with path.open('rb') as source:
                    archive.addfile(info, source)
    except BaseException as exc:
        producer_error = exc
        process.kill()
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        consumer.join()
        returncode = process.wait()
        process.stdout.close()
    if errors:
        raise errors[0]
    if producer_error:
        raise producer_error
    if returncode:
        raise RuntimeError('Archive compression failed')
    if part_bytes is not None:
        verify_parts(parts, expected)
        records = []
        for path in parts:
            final = path.with_suffix('')
            if final.exists():
                raise ValueError('Preserve existing final part')
            path.replace(final)
            records.append(dict(path=final.name, bytes=final.stat().st_size, sha256=sha256(final)))
        record = dict(path=target.name, bytes=sum(r['bytes'] for r in records),
                      sha256=whole_hash.hexdigest(), members=expected, parts=records)
        verified_asset_paths(target.parent, record)
        return record
    verify_members(temporary, expected)
    temporary.replace(target)
    return dict(path=target.name, bytes=target.stat().st_size, sha256=sha256(target), members=expected)
