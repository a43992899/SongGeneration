rm -rf output/probe.GTZANGenre.LeVo.pre_vq
python cli.py fit -c probe.LeVo.GTZANGenre.yaml
python cli.py test -c probe.LeVo.GTZANGenre.yaml