python awq/tinyllama_my_bfp_hellaswag.py --bft --m_bit 2 --b_size 128 | tee awq/log/bft_m2_b128_hellaswag.log
python awq/tinyllama_my_bfp_hellaswag.py --bft --m_bit 3 --b_size 128 | tee awq/log/bft_m3_b128_hellaswag.log
python awq/tinyllama_my_bfp_hellaswag.py --bft --m_bit 4 --b_size 128 | tee awq/log/bft_m4_b128_hellaswag.log
python awq/tinyllama_my_bfp_hellaswag.py --bft --m_bit 5 --b_size 128 | tee awq/log/bft_m5_b128_hellaswag.log