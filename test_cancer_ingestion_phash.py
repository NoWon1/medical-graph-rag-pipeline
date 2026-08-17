import unittest
from PIL import Image
import numpy as np

# We also need to import the related module and variables so that we can patch them
import cancer_ingestion
from cancer_ingestion import _compute_phash, _phash_hamming, _is_in_phash_blocklist, PHASH_HAMMING_THRESH

class TestPHashFunctions(unittest.TestCase):
    def setUp(self):
        # Clear the blocklist before each test
        cancer_ingestion._phash_blocklist.clear()

    def test_compute_phash_basic(self):
        # Create a simple checkerboard pattern
        img_array = np.zeros((16, 16), dtype=np.uint8)
        img_array[:8, :8] = 255
        img_array[8:, 8:] = 255
        img = Image.fromarray(img_array)

        phash = _compute_phash(img, hash_size=8)
        self.assertIsInstance(phash, int)
        self.assertGreaterEqual(phash, 0)

    def test_compute_phash_identical_images(self):
        # Create a random image
        img_array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        img = Image.fromarray(img_array)

        phash1 = _compute_phash(img)
        phash2 = _compute_phash(img)

        self.assertEqual(phash1, phash2)

    def test_phash_hamming_identical(self):
        h1 = 0b10101010
        h2 = 0b10101010
        self.assertEqual(_phash_hamming(h1, h2), 0)

    def test_phash_hamming_different(self):
        h1 = 0b11110000
        h2 = 0b00001111
        # In a 8-bit number, if they are exact opposites, distance is 8
        self.assertEqual(_phash_hamming(h1, h2), 8)

        h3 = 0b11110000
        h4 = 0b11100000
        # Only 1 bit difference
        self.assertEqual(_phash_hamming(h3, h4), 1)

    def test_is_in_phash_blocklist_match(self):
        # Add a known hash to the blocklist
        known_hash = 0b1111000011110000
        cancer_ingestion._phash_blocklist.append(known_hash)

        # Exact match
        self.assertTrue(_is_in_phash_blocklist(known_hash))

        # Close match (within threshold) - Assuming threshold is at least 1
        # Change 1 bit (distance 1)
        close_hash = known_hash ^ 1
        self.assertTrue(_is_in_phash_blocklist(close_hash))

    def test_is_in_phash_blocklist_no_match(self):
        known_hash = 0b1111000011110000
        cancer_ingestion._phash_blocklist.append(known_hash)

        # Far hash (all bits flipped)
        far_hash = ~known_hash & ((1 << 64) - 1)  # Assuming a 64-bit mask for 8x8 pHash if needed, but simple bitwise flip works for checking high distance
        # Actually just construct a very different one
        far_hash = 0b0000111100001111

        # If threshold is small, distance of 8 will be False
        # Let's ensure it exceeds PHASH_HAMMING_THRESH
        if 8 > PHASH_HAMMING_THRESH:
            self.assertFalse(_is_in_phash_blocklist(far_hash))

    def test_compute_phash_edge_cases(self):
        # Extremely small image (e.g., 1x1) - should still work due to resize in _compute_phash
        img_array = np.array([[255]], dtype=np.uint8)
        img = Image.fromarray(img_array)
        phash = _compute_phash(img, hash_size=8)
        self.assertIsInstance(phash, int)

        # Extremely large solid color image
        img_array = np.zeros((1000, 1000), dtype=np.uint8)
        img = Image.fromarray(img_array)
        phash_solid = _compute_phash(img, hash_size=8)

        # Solid color image should yield a specific hash (probably 0 because all pixels are same, so left > right is never true)
        self.assertEqual(phash_solid, 0)

if __name__ == '__main__':
    unittest.main()
