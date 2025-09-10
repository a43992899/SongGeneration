rm -rf output/probe.GS.LeVo.vq_emb
python cli.py fit -c probe.LeVo.GS.yaml
python cli.py test -c probe.LeVo.GS.yaml