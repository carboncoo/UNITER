import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    result_dir = '/data/share/UNITER/ve/da/pos/seed2/GloVe/txt_db/ve_train.db/results_test/results_4000_all.json'
    results = json.load(open(result_dir))

    answers = [float(res['answer']) for res in results]

    # plot
    y = np.array(answers)
    plt.hist(y, bins=400, histtype='step')
    plt.savefig('answers')

if __name__ == '__main__':
    main()