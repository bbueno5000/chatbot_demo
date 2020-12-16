"""
Trains a memory network on the bAbI dataset.

References:
- Jason Weston, Antoine Bordes, Sumit Chopra, Tomas Mikolov, Alexander M. Rush,
  "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks",
  http://arxiv.org/abs/1502.05698
- Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus,
  "End-To-End Memory Networks",
  http://arxiv.org/abs/1503.08895

Reaches 98.6% accuracy on task 'single_supporting_fact_10k' after 120 epochs.
Time per epoch: 3s on CPU (core i7).
"""
import functools
import keras
import numpy
import re
import tarfile

class Chatbot:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        try:
            path = keras.utils.data_utils.get_file(
                'babi-tasks-v1-2.tar.gz',
                origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise
        tar = tarfile.open(path)
        challenges = {
            'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
            'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'}
        challenge_type = 'single_supporting_fact_10k'
        challenge = challenges[challenge_type]
        print('Extracting stories for the challenge:', challenge_type)
        train_stories = self.get_stories(tar.extractfile(challenge.format('train')))
        test_stories = self.get_stories(tar.extractfile(challenge.format('test')))
        vocab = set()
        for story, q, answer in train_stories + test_stories:
            vocab |= set(story + q + [answer])
        vocab = sorted(vocab)
        vocab_size = len(vocab) + 1
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
        print('-')
        print('Vocab size:', vocab_size, 'unique words')
        print('Story max length:', story_maxlen, 'words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-')
        print('Here\'s what a "story" tuple looks like (input, query, answer):')
        print(train_stories[0])
        print('-')
        print('Vectorizing the word sequences...')
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        inputs_train, queries_train, answers_train = self.vectorize_stories(
            train_stories, word_idx, story_maxlen, query_maxlen)
        inputs_test, queries_test, answers_test = self.vectorize_stories(
            test_stories, word_idx, story_maxlen, query_maxlen)
        print('-')
        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', inputs_train.shape)
        print('inputs_test shape:', inputs_test.shape)
        print('-')
        print('queries: integer tensor of shape (samples, max_length)')
        print('queries_train shape:', queries_train.shape)
        print('queries_test shape:', queries_test.shape)
        print('-')
        print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
        print('answers_train shape:', answers_train.shape)
        print('answers_test shape:', answers_test.shape)
        print('-')
        print('Compiling...')
        input_sequence = keras.layers.Input((story_maxlen,))
        question = keras.layers.Input((query_maxlen,))
        input_encoder_m = keras.models.Sequential()
        input_encoder_m.add(keras.layers.embeddings.Embedding(
            input_dim=vocab_size, output_dim=64))
        input_encoder_m.add(keras.layers.Dropout(0.3))
        input_encoder_c = keras.models.Sequential()
        input_encoder_c.add(keras.layers.embeddings.Embedding(
            input_dim=vocab_size, output_dim=query_maxlen))
        input_encoder_c.add(keras.layers.Dropout(0.3))
        question_encoder = keras.models.Sequential()
        question_encoder.add(keras.layers.embeddings.Embedding(
            input_dim=vocab_size, output_dim=64, input_length=query_maxlen))
        question_encoder.add(keras.layers.Dropout(0.3))
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)
        match = keras.layers.dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = keras.layers.Activation('softmax')(match)
        response = keras.layers.add([match, input_encoded_c])
        response = keras.layers.Permute((2, 1))(response)
        answer = keras.layers.concatenate([response, question_encoded])
        answer = keras.layers.LSTM(32)(answer)
        answer = keras.layers.Dropout(0.3)(answer)
        answer = keras.layers.Dense(vocab_size)(answer)
        answer = keras.layers.Activation('softmax')(answer)
        model = keras.models.Model([input_sequence, question], answer)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(
            [inputs_train, queries_train], answers_train, batch_size=32,
            epochs=120, validation_data=([inputs_test, queries_test], answers_test))

    def get_stories(self, f, only_supporting=False, max_length=None):
        """
        Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        """
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: functools.reduce(lambda x, y: x + y, data)
        data = [
            (flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
        return data

    def parse_stories(self, lines, only_supporting=False):
        """
        Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        """
        data, story = list(), list()
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            if nid == 1:
                story = list()
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                substory = None
                if only_supporting:
                    supporting = map(int, supporting.split())
                    substory = [story[i - 1] for i in supporting]
                else:
                    substory = [x for x in story if x]
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
        return data

    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        """
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

    def vectorize_stories(self, data, word_idx, story_maxlen, query_maxlen):
        """
        TODO: docstring
        """
        X, Xq, Y = list(), list(), list()
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            y = numpy.zeros(len(word_idx) + 1)
            y[word_idx[answer]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
        return (
            keras.preprocessing.sequence.pad_sequences(X, maxlen=story_maxlen),
            keras.preprocessing.sequence.pad_sequences(Xq, maxlen=query_maxlen), numpy.array(Y))

if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot()
