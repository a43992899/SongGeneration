rm -rf output/probe.GTZANBeatTracking.LeVo.vq_emb
python cli.py fit -c probe.LeVo.GTZANBeatTracking.yaml
python cli.py test -c probe.LeVo.GTZANBeatTracking.yaml