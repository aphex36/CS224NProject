To generate vocab:
python3 generate_vocab.py trainfile testfile
ex: python3 generate_vocab.py smaller_train_shuffled.json smaller_test_shuffled.json

To test a model:
python3 run.py {train, test} {trainfile, testfile} {bnn, cnn, rnn} model_savefile {cpu, cuda}
ex: python3 run.py train smaller_train_shuffled.json bnn bnn_model.bin cpu

NOTES:
If using a different trainfile / testfile than before, be sure to delete pkl cache:
rm *.pkl

TMUX:
to set up a session, run code, and exit:

tmux new -s <session name>
<my command (eg. python 3 run.py...)>
tmux detach

to reattach:
tmux a -t <session name>