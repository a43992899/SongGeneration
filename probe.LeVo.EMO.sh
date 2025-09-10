rm -rf output/probe.EMO.LeVo.vq_emb
python cli.py fit -c probe.LeVo.EMO.yaml
python cli.py test -c probe.LeVo.EMO.yaml