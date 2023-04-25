from promcse import PromCSE


train_dataset = open('data/train.json', 'r', encoding='utf-8').readlines()
validation_dataset = open('data/dev.json', 'r', encoding='utf-8').readlines()
test_dataset = open('data/test.json', 'r', encoding='utf-8').readlines()
### for unsupervised promcse-bert-base-uncased
model_name_or_path = 'YuxinJiang/unsup-promcse-bert-base-uncased'
pooler_type = 'cls_before_pooler'
pre_seq_len = 16

# ### for supervised promcse-roberta-base
# model_name_or_path = 'YuxinJiang/sup-promcse-roberta-base'
# pooler_type = 'cls'
# pre_seq_len = 10

# ### for supervised promcse-roberta-large
# model_name_or_path = 'YuxinJiang/sup-promcse-roberta-large'
# pooler_type = 'cls'
# pre_seq_len = 10

model = PromCSE(model_name_or_path=model_name_or_path,
          pooler_type=pooler_type,
          pre_seq_len=pre_seq_len)



for data in test_dataset:
    data = eval(data)
    claim = data['claim']
    evidences = data['evidences']
    for evidence in evidences:
        similarities = model.similarity(claim, evidence)
        print("\n\n", similarities)

# example_sentences = [
#     'An animal is biting a persons finger.',
#     'A woman is reading.',
#     'A man is lifting weights in a garage.',
#     'A man plays the violin.',
#     'A man is eating food.',
#     'A man plays the piano.',
#     'A panda is climbing.',
#     'A man plays a guitar.',
#     'A woman is slicing a meat.',
#     'A woman is taking a picture.'
# ]
#
# example_queries = [
#     'A man is playing music.',
#     'A woman is making a photo.'
# ]
#
# print("\n=========Naive brute force search============\n")
# model.build_index(example_sentences, use_faiss=False)
# results = model.search(example_queries)
# for i, result in enumerate(results):
#     print("Retrieval results for query: {}".format(example_queries[i]))
#     for sentence, score in result:
#         print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
#     print("")
#
# print("\n=========Search with Faiss backend============\n")
# model.build_index(example_sentences, use_faiss=True)
# results = model.search(example_queries)
# for i, result in enumerate(results):
#     print("Retrieval results for query: {}".format(example_queries[i]))
#     for sentence, score in result:
#         print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
#     print("")