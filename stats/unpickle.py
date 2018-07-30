import pickle

print("gru glove 150")
with open('gru_glove_epoch_5_150d_f1s.p', 'rb') as f1:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f1)
    print("f1")
    print(data)

with open('gru_glove_epoch_5_150d_nbatch.p', 'rb') as f2:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f2)
	print("nbatch")
	print(data)

with open('gru_glove_epoch_5_150d_precisions.p', 'rb') as f3:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f3)
	print("precision")
	print(data)

with open('gru_glove_epoch_5_150d_recalls.p', 'rb') as f4:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f4)
	print("recalls")
	print(data)

print("gru glove 50")
with open('gru_glove_epoch_5_f1s.p', 'rb') as f5:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f5)
    print("f1")
    print(data)

with open('gru_glove_epoch_5_nbatch.p', 'rb') as f6:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f6)
	print("nbatch")
	print(data)

with open('gru_glove_epoch_5_precisions.p', 'rb') as f7:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f7)
	print("precision")
	print(data)

with open('gru_glove_epoch_5_recalls.p', 'rb') as f8:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f8)
	print("recalls")
	print(data)

print("lstm glove 150")
with open('lstm_glove_epoch_5_150d_f1s.p', 'rb') as f9:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f9)
    print("f1")
    print(data)

with open('lstm_glove_epoch_5_150d_nbatch.p', 'rb') as f10:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f10)
	print("nbatch")
	print(data)

with open('lstm_glove_epoch_5_150d_precisions.p', 'rb') as f11:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f11)
	print("precision")
	print(data)

with open('lstm_glove_epoch_5_150d_recalls.p', 'rb') as f12:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f12)
	print("recalls")
	print(data)

print("lstm glove 50")
with open('lstm_glove_epoch_5_f1s.p', 'rb') as f13:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f13)
    print("f1")
    print(data)

with open('lstm_glove_epoch_5_nbatch.p', 'rb') as f14:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f14)
	print("nbatch")
	print(data)

with open('lstm_glove_epoch_5_precisions.p', 'rb') as f15:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f15)
	print("precision")
	print(data)

with open('lstm_glove_epoch_5_recalls.p', 'rb') as f16:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f16)
	print("recalls")
	print(data)

print("lstm word2vec 50")
with open('lstm_word2vec_epoch_5_50d_f1s.p', 'rb') as f17:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
    data = pickle.load(f17)
    print("f1")
    print(data)

with open('lstm_word2vec_epoch_5_50d_nbatch.p', 'rb') as f18:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f18)
	print("nbatch")
	print(data)

with open('lstm_word2vec_epoch_5_50d_precisions.p', 'rb') as f19:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f19)
	print("precision")
	print(data)

with open('lstm_word2vec_epoch_5_50d_recalls.p', 'rb') as f20:
	# The protocol version used is detected automatically, so we do not
	# have to specify it.
	data = pickle.load(f20)
	print("recalls")
	print(data)