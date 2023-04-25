import json

datasets = json.load(open('datasets/train.json', 'r', encoding='utf8'))
save = open('datasets/train.txt', 'w', encoding='utf8')
save2 = open('datasets/train.data', 'w', encoding='utf8')

# 寫入資料
sentence = []
for data in datasets:
    claim = data['claim']
    # evidences = [data['gold evidence'][str(i)]['text'] for i in range(5)]
    sentence.append(claim)
    # sentence.append(evidences[0])
    if len(sentence) == 10000:
        break
for sent in sentence:
  save.write(sent+'\n')
  save2.write(sent + '\n')
save.close()
save2.close()