rm -rf output/probe.Chords1217.LeVo.vq_emb
python cli.py fit -c probe.LeVo.Chords1217.yaml
python cli.py test -c probe.LeVo.Chords1217.yaml