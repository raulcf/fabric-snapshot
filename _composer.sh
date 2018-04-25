echo "v50 static"
python run_relemb_composition.py --we_model /data/kevin21/testing/mitall/allvectors/mitall_v50_n100_i8.txt --dataset /data/raulcf/datasets/mitdwhdataall/ --output /data/raulcf/relemb/mitdwh/mitall_v50_n100_i8_static.pkl
echo "v50 dynamic"
python run_relemb_composition.py --we_model /data/kevin21/testing/mitall/allvectors/mitall_v50_n100_i8_csv.txt --dataset /data/raulcf/datasets/mitdwhdataall --output /data/raulcf/relemb/mitdwh/mitall_v50_n100_i8_dynamic.pkl
echo "v100 static"
python run_relemb_composition.py --we_model /data/kevin21/testing/mitall/allvectors/mitall_v100_n100_i8.txt --dataset /data/raulcf/datasets/mitdwhdataall --output /data/raulcf/relemb/mitdwh/mitall_v100_n100_i8_static.pkl
echo "v100 dynamic"
python run_relemb_composition.py --we_model /data/kevin21/testing/mitall/allvectors/mitall_v100_n100_i8_csv.txt --dataset /data/raulcf/datasets/mitdwhdataall --output /data/raulcf/relemb/mitdwh/mitall_v100_n100_i8_dynamic.pkl
echo "v300 static"
python run_relemb_composition.py --we_model /data/kevin21/testing/mitall/allvectors/mitall_v300_n100_i8.txt --dataset /data/raulcf/datasets/mitdwhdataall --output /data/raulcf/relemb/mitdwh/mitall_v300_n100_i8_static.pkl
echo "v300 dynamic"
python run_relemb_composition.py --we_model /data/kevin21/testing/mitall/allvectors/mitall_v300_n100_i8_csv.txt --dataset /data/raulcf/datasets/mitdwhdataall --output /data/raulcf/relemb/mitdwh/mitall_v300_n100_i8_dynamic.pkl
