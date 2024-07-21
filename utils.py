import tqdm


class TranslationUtils:
    @staticmethod
    def gen_index_table(index_path, train_path, total_length, desc="索引表"):
        with open(index_path, encoding="UTF-8", mode='w') as index_file, \
                open(train_path, encoding="utf-8", mode='r') as train_file:
            pos = 0
            for line in tqdm.tqdm(train_file, total=total_length, desc=desc):
                index_file.write("%s\n" % pos)
                pos += len(line.encode('utf-8'))