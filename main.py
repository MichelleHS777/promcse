from promcse import PromCSE

test_dataset = open('./datasets/preprocessed/test.json', 'r', encoding='utf-8')
model = PromCSE("result/my-unsup-promcse-bert-base-chinese", "cls_before_pooler", 16)

for data in test_dataset:
    data = eval(data)
    claim = data['claim']
    evidences = data['evidences']
    similarities = model.similarity(claim, evidences)
    sent2sim = list(similarities)
    print(sent2sim)
    # print("\n\n", similarities)

    print("\n=========Naive brute force search============\n")
    model.build_index(evidences, use_faiss=False)
    results = model.search(claim)
    sent = []
    for result in enumerate(results):
        print("Retrieval results for query: {}".format(claim))

    # print("\n=========Search with Faiss backend============\n")
    # model.build_index(evidences, use_faiss=True)
    # results = model.search(claim)
    # for i, result in enumerate(results):
    #     print("Retrieval results for query: {}".format(claim[i]))
    #     for sentence, score in result:
    #         print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
    #     print("")
