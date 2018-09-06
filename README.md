# JPEG Eigen-algorithm features extractor
*N. Bonettini, L. Bondi, P. Bestagini, S. Tubaro,
"JPEG Implementation Forensics Based on Eigen-Algorithms",
IEEE International Workshop on Information Forensics and Security (WIFS),
2018*


## Functions

### jpeg_recompress_pil
Re-compress a JPEG image using the same quantization matrix and PIL implementation

### jpeg_feature
Extract JPEG eigenfeatures

## Test
```bash
python3 -m unittest test_extractor.TestExtractor
```