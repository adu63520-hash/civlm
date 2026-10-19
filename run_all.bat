@echo off
chcp 65001 >nul
cd /d %~dp0
python gen_data.py || goto :err
python eval_icl.py --baselines_only || goto :err
python pretrain.py || goto :err
python eval_icl.py --ckpt out/civlm26m/best.pt || goto :err
python eval_icl.py --ckpt_dir out/civlm26m/ckpts --ks 0 4 12 --styles structured --n_worlds 30 || goto :err
echo.
echo 全部完成，结果在 results\ 目录
pause
exit /b 0
:err
echo 出错了，请看上面的报错信息
pause
