rm -rf output/probe.GTZANBeatTracking.LeVo.vq_emb.100hz
python cli.py fit -c probe.LeVo.GTZANBeatTracking.100hz.yaml
python cli.py test -c probe.LeVo.GTZANBeatTracking.100hz.yaml