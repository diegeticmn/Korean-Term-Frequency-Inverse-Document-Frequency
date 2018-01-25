from gensim import models
from gensim import corpora
from konlpy.tag import Twitter; t = Twitter()
import string

from konlpy.corpus import kobill

with open('vegetariankoreantrimmed.txt', 'r') as myfile:
    veg=myfile.read().replace('\n', '')

docs_ko = [veg] + [kobill.open(i).read() for i in kobill.fileids()]

translator = str.maketrans('', '', string.punctuation)

nopunc = [doc.translate(translator) for doc in docs_ko]

pos = lambda d: ['/'.join(p) for p in t.pos(d, stem=True, norm=True)]
texts_ko = [pos(doc) for doc in nopunc]

dictionary_ko = corpora.Dictionary(texts_ko)

tf_ko = [dictionary_ko.doc2bow(text) for text in texts_ko]
tfidf_model_ko = models.TfidfModel(tf_ko)
tfidf_ko = tfidf_model_ko[tf_ko]

# print top 25 elements of first document's tf-idf vector
top = (sorted(tfidf_ko.corpus[0], key=lambda x: x[1], reverse=True)[:105])
print(top)
# print token of most frequent element
toplist = [x[0] for x in top]
for i in toplist:
	print(dictionary_ko.get(i))
