#!/usr/bin/env python3
"""
Contains classes and functions to perform rsync-style resumable, recursive downloading of the individual files in a hail table file structure on Google Cloud.
Needed because downloading the entire table at once using the standard approach was not working. This is far slower, but more robust.
"""
from typing import List, Dict, Any
from hailtop.fs.stat_result import FileType
import os
from multiprocessing import Pool


NUM_PROCESSES = 16  # Define the number of processes to use
RPC = "velia-genebass"  # Name of google cloud project to charge to


def get_listing(remote_path: str, requester_pays_config=RPC) -> List[Dict[str, Any]]:
    return hfs.ls(remote_path, requester_pays_config=requester_pays_config)


def copy_file(src_path: str, dest_fpath: str, requester_pays_config=RPC) -> None:
    hfs.copy(src_path, dest_fpath, requester_pays_config=requester_pays_config)


def tab_print(msg, depth):
    print("\t" * depth + msg)


def generate_log_func(silent):
    if silent:
        return lambda msg: None
    else:
        return lambda msg: tab_print(msg, depth=0)


def is_file_newer(remote_fle, local_fpath):
    try:
        local_stat = os.stat(local_fpath)
    except FileNotFoundError:
        return True
    return (
        remote_fle.size != local_stat.st_size
        or remote_fle.modification_time > local_stat.st_mtime
    )


def copy_file_wrapper(args):
    print(args)
    src_path, dest_fpath, requester_pays_config = args
    copy_file(src_path, dest_fpath, requester_pays_config)


def copy_recursively_breadth_first(
    remote_base_path: str,
    local_base_path: str,
    requester_pays_config=RPC,
    silent=True,
    sync_mode=True,
):
    queue = [(remote_base_path, local_base_path)]
    missing_files = []
    downloaded_file_count = 0
    skipped_file_count = 0
    directories_traversed_count = 0
    batch_count = 0

    log_func = generate_log_func(silent=silent)

    with Pool(NUM_PROCESSES) as pool:
        while queue:
            batch_count += 1
            if batch_count % 100 == 0:
                print(f"Processing batch {batch_count} ...")

            current_path, local_path = queue.pop(0)
            os.makedirs(local_path, exist_ok=True)
            log_func(f"\tGetting listing for {current_path} ...")

            try:
                listing = get_listing(
                    current_path, requester_pays_config=requester_pays_config
                )
            except FileNotFoundError:
                log_func(f"\tFile not found: {current_path}!")
                missing_files.append(current_path)
                continue

            log_func(f"\tGot listing for {current_path}.")
            directories = []
            files_to_copy = []

            for fle in listing:
                fname = fle.path.split("/")[-1]

                log_func(f"\tExamining {fle.path}...")
                local_fpath = os.path.join(local_path, fname)
                remote_fpath = fle.path

                if fle.typ == FileType.DIRECTORY:
                    directories.append((remote_fpath, local_fpath))
                    log_func(f"\tAdding {remote_fpath} to queue...")
                    directories_traversed_count += 1

                elif fle.typ == FileType.FILE:
                    log_func(
                        f"\tChecking if {remote_fpath} is newer than {local_fpath} ..."
                    )
                    if (
                        is_file_newer(remote_fle=fle, local_fpath=local_fpath)
                        or not sync_mode
                    ):
                        log_func(f"\t{remote_fpath} is newer, adding to copy queue")
                        files_to_copy.append(
                            (remote_fpath, local_fpath, requester_pays_config)
                        )
                    else:
                        log_func(f"\t{remote_fpath} is older.")
                        skipped_file_count += 1

            # Process files in parallel
            log_func(f"\tCopying {len(files_to_copy)} files ...")
            # pool.map(copy_file_wrapper, files_to_copy)
            for remote_fpath, local_fpath, requester_pays_config in files_to_copy:
                log_func(f"\tCopying {remote_fpath} to {local_fpath} ...")
                copy_file(
                    src_path=remote_fpath,
                    dest_fpath=local_fpath,
                    requester_pays_config=requester_pays_config,
                )
                downloaded_file_count += 1

            # Add directories to the queue to process them in the next iterations
            queue.extend(directories)
            log_func(f"Queue now has {len(queue)} elements.")

    print(
        f"All done. Traversed {directories_traversed_count}, downloaded {downloaded_file_count} files, skipped {skipped_file_count}, {len(missing_files)} files were missing."
    )

    return missing_files
