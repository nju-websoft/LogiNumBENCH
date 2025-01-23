from tqdm import tqdm
from LogiNumBench import Theory, RandomGene
import json
import os

tem_prepath = './templated-data/LD{level}/'

for configType in ['test', 'train', 'val']:
    with open(f"configs/{configType}_config.json") as jsf:
        obj = json.load(jsf)
        RandomGene.load_config(obj)

    params = [[4, 4, 6, 6, 3], [4, 4, 6, 6, 3], [4, 4, 6, 6, 3],
              [5, 6, 7, 7, 4], [5, 6, 8, 8, 5], [6, 7, 8, 10, 5]]
    for level in range(1, 7):
        prepath = tem_prepath.format(level=level) + f"{configType}_config/"
        if not os.path.exists(prepath):
            os.makedirs(prepath)
        para = params[level - 1]

        with open(prepath + 'data.txt', 'w', encoding='utf-8') as f, \
                open(prepath + "datanl.txt", 'w', encoding='utf-8') as p, \
                open(prepath + 'datajs.jsonl', 'w') as jsf:
            for i in tqdm(range(20000 if configType == 'train' else 2000),
                          desc='Generating...'):
                while True:
                    theory = Theory(i,
                                    entityNum=para[0],
                                    attrNum=para[1],
                                    factNum=para[2],
                                    ruleNum=para[3],
                                    relationNum=para[4],
                                    depth=level)
                    if theory.assertion is not None:
                        break
                f.write(str(theory))
                f.write("\n")
                p.write(theory.nl())
                p.write("\n")
                jsf.write(json.dumps(theory.to_json()))
                jsf.write('\n')
