"""
Test file distribution logic for Qwen3ChatCompletionParquetDataset in multi-process, multi-worker scenarios

Validation points:
1. Each file is processed by only one worker (no duplication)
2. All files are processed (no omission)
3. Works correctly under different rank and worker combinations
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys


class TestFileDistribution(unittest.TestCase):
    """Test file distribution logic"""

    def setUp(self):
        """Set up test environment"""
        # Create mock file list
        self.data_files = [
            (f"file_{i}.parquet", 0) for i in range(100)  # 100 files, epoch=0
        ]
        self.num_workers = 4
    
    def _get_file_distribution(self, rank, world_size, worker, num_workers):
        """
        Simulate file distribution logic, return file indices for this worker

        Args:
            rank: Process rank
            world_size: Total number of processes
            worker: Worker ID
            num_workers: Number of workers per process

        Returns:
            list: File index list
        """
        total_num_workers = num_workers * world_size
        local_worker_idx = rank * num_workers + worker
        fn_list = [
            idx for idx, fn in enumerate(self.data_files) 
            if idx % total_num_workers == local_worker_idx
        ]
        return fn_list
    
    def test_file_distribution_no_overlap(self):
        """Test file distribution without overlap: each file is processed by only one worker"""
        world_size = 2
        num_workers = 4
        
        # Collect files assigned to all workers
        all_assigned_files = set()

        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_files = self._get_file_distribution(rank, world_size, worker, num_workers)
                file_indices = set(assigned_files)

                # Check for overlap
                overlap = all_assigned_files & file_indices
                self.assertEqual(
                    len(overlap), 0,
                    f"Rank {rank}, Worker {worker} assigned files overlap with existing assignments: {overlap}"
                )

                all_assigned_files.update(file_indices)

        # Verify all files are assigned
        total_files = len(self.data_files)
        self.assertEqual(
            len(all_assigned_files), total_files,
            f"File assignment incomplete: expected {total_files} files, actually assigned {len(all_assigned_files)}"
        )
    
    def test_file_distribution_completeness(self):
        """Test file distribution completeness: all files are processed"""
        world_size = 2
        num_workers = 4

        all_assigned_files = set()

        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_files = self._get_file_distribution(rank, world_size, worker, num_workers)
                all_assigned_files.update(assigned_files)

        # Verify all files are assigned
        expected_files = set(range(len(self.data_files)))
        self.assertEqual(
            all_assigned_files, expected_files,
            f"File assignment incomplete: missing files {expected_files - all_assigned_files}"
        )
    
    def test_file_distribution_different_configs(self):
        """Test file distribution under different configurations"""
        test_configs = [
            (1, 1),   # Single process, single worker
            (1, 4),   # Single process, 4 workers
            (2, 2),   # 2 processes, 2 workers each
            (4, 2),   # 4 processes, 2 workers each
            (2, 8),   # 2 processes, 8 workers each
        ]
        
        for world_size, num_workers in test_configs:
            with self.subTest(world_size=world_size, num_workers=num_workers):
                all_assigned_files = set()
                
                for rank in range(world_size):
                    for worker in range(num_workers):
                        assigned_files = self._get_file_distribution(
                            rank, world_size, worker, num_workers
                        )
                        file_indices = set(assigned_files)

                        # Check for overlap
                        overlap = all_assigned_files & file_indices
                        self.assertEqual(
                            len(overlap), 0,
                            f"Config (world_size={world_size}, num_workers={num_workers}), "
                            f"Rank {rank}, Worker {worker} has overlap: {overlap}"
                        )

                        all_assigned_files.update(file_indices)

                # Verify completeness
                expected_files = set(range(len(self.data_files)))
                self.assertEqual(
                    all_assigned_files, expected_files,
                    f"Config (world_size={world_size}, num_workers={num_workers}) "
                    f"file assignment incomplete: missing {expected_files - all_assigned_files}"
                )
    
    def test_file_distribution_balance(self):
        """Test file distribution load balancing (each worker should be assigned roughly equal number of files)"""
        world_size = 2
        num_workers = 4
        total_workers = world_size * num_workers

        file_counts = []
        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_files = self._get_file_distribution(rank, world_size, worker, num_workers)
                file_counts.append(len(assigned_files))

        # Calculate expected file count (should be roughly equal)
        expected_per_worker = len(self.data_files) / total_workers
        min_files = int(expected_per_worker)
        max_files = int(expected_per_worker) + 1

        # Verify each worker's file count is within reasonable range
        for count in file_counts:
            self.assertGreaterEqual(count, min_files, "Too few files assigned")
            self.assertLessEqual(count, max_files, "Too many files assigned")

        # Verify total count is correct
        self.assertEqual(
            sum(file_counts), len(self.data_files),
            f"Total file count mismatch: expected {len(self.data_files)}, actual {sum(file_counts)}"
        )
    
    def test_file_distribution_with_epochs(self):
        """Test file distribution with multiple epochs"""
        # Create multi-epoch file list
        data_files_multi_epoch = []
        for epoch in range(3):
            for i in range(20):
                data_files_multi_epoch.append((f"file_{i}.parquet", epoch))

        self.data_files = data_files_multi_epoch

        world_size = 2
        num_workers = 4

        # Collect assignments by (file_idx, epoch)
        all_assigned = set()

        for rank in range(world_size):
            for worker in range(num_workers):
                assigned_indices = self._get_file_distribution(
                    rank, world_size, worker, num_workers
                )
                # Convert indices to (filename, epoch) tuples
                for idx in assigned_indices:
                    file_name, epoch = self.data_files[idx]
                    all_assigned.add((file_name, epoch))

        # Verify all (file, epoch) combinations are assigned
        expected = set((fn, ep) for fn, ep in self.data_files)
        self.assertEqual(
            all_assigned, expected,
            f"Multi-epoch file assignment incomplete: missing {expected - all_assigned}"
        )


class TestFileDistributionLogic(unittest.TestCase):
    """Test core algorithm of file distribution logic"""

    def setUp(self):
        """Set up test environment"""
        self.data_files = [
            (f"file_{i}.parquet", 0) for i in range(50)
        ]

    def test_distribution_algorithm(self):
        """Test correctness of file distribution algorithm"""
        # Simulate distribution logic in Qwen3NaiveParquetDataset.__iter__local_shuffle
        rank = 0
        world_size = 2
        worker = 0
        num_workers = 2

        total_num_workers = num_workers * world_size
        local_worker_idx = rank * num_workers + worker
        fn_list = [
            fn for idx, fn in enumerate(self.data_files)
            if idx % total_num_workers == local_worker_idx
        ]

        # Verify file list is not empty
        self.assertGreater(len(fn_list), 0, "File list should not be empty")

        # Verify file indices are correct
        expected_indices = [
            idx for idx in range(len(self.data_files))
            if idx % total_num_workers == local_worker_idx
        ]
        actual_indices = [
            idx for idx, fn in enumerate(self.data_files) if fn in fn_list
        ]
        self.assertEqual(
            set(actual_indices), set(expected_indices),
            "File index assignment is incorrect"
        )


def run_distribution_test_manual():
    """
    Manually run file distribution test, print detailed assignment information
    For debugging and verification
    """
    print("=" * 80)
    print("File Distribution Test - Manual Verification")
    print("=" * 80)

    # Test configurations
    data_files = [(f"file_{i}.parquet", 0) for i in range(100)]
    test_configs = [
        (1, 1, "Single process, single worker"),
        (1, 4, "Single process, 4 workers"),
        (2, 2, "2 processes, 2 workers each"),
        (4, 2, "4 processes, 2 workers each"),
        (2, 8, "2 processes, 8 workers each"),
    ]
    
    for world_size, num_workers, desc in test_configs:
        print(f"\nConfig: {desc} (world_size={world_size}, num_workers={num_workers})")
        print("-" * 80)

        total_num_workers = num_workers * world_size
        all_assigned = {}

        for rank in range(world_size):
            for worker in range(num_workers):
                local_worker_idx = rank * num_workers + worker
                assigned_files = [
                    idx for idx, fn in enumerate(data_files)
                    if idx % total_num_workers == local_worker_idx
                ]
                all_assigned[(rank, worker)] = assigned_files

                print(f"  Rank {rank}, Worker {worker} (local_idx={local_worker_idx}): "
                      f"{len(assigned_files)} files, index range: {min(assigned_files) if assigned_files else 'N/A'}-{max(assigned_files) if assigned_files else 'N/A'}")

        # Verify completeness
        all_file_indices = set()
        for assigned in all_assigned.values():
            all_file_indices.update(assigned)

        expected_indices = set(range(len(data_files)))
        missing = expected_indices - all_file_indices
        extra = all_file_indices - expected_indices

        if missing:
            print(f"  X Missing file indices: {sorted(missing)}")
        if extra:
            print(f"  X Extra file indices: {sorted(extra)}")
        if not missing and not extra:
            print(f"  OK File assignment complete: all {len(data_files)} files correctly assigned")

        # Check for overlap
        has_overlap = False
        for (r1, w1), files1 in all_assigned.items():
            for (r2, w2), files2 in all_assigned.items():
                if (r1, w1) >= (r2, w2):  # Avoid duplicate checks
                    continue
                overlap = set(files1) & set(files2)
                if overlap:
                    print(f"  X Overlap detected: Rank {r1}, Worker {w1} and Rank {r2}, Worker {w2} overlap files: {sorted(overlap)}")
                    has_overlap = True

        if not has_overlap:
            print(f"  OK No overlap: all files processed by only one worker")


if __name__ == '__main__':
    # Run unit tests
    print("Running unit tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)

    # Run manual verification
    print("\n" + "=" * 80)
    run_distribution_test_manual()

