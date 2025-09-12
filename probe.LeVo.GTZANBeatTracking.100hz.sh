rm -rf output/probe.GTZANBeatTracking.LeVo.pre_vq.100hz
python cli.py fit -c probe.LeVo.GTZANBeatTracking.100hz.yaml
python cli.py test -c probe.LeVo.GTZANBeatTracking.100hz.yaml