$env:KMP_DUPLICATE_LIB_OK = "TRUE"
python smoke_test.py --model glf_tiny --img-size 64 --batch-size 2 --skip-flops
python smoke_test.py --model cnn_only --img-size 64 --batch-size 2 --skip-flops
python smoke_test.py --model mobilevit_lite --img-size 64 --batch-size 2 --attention mha --skip-flops
