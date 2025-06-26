#source ~/.bashrc
#
rm -r train val test full
mkdir train val test full
python make_rxn.py
