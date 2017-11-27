import os


class Document(object):
    def __init__(self, name, header, content, is_spam=None):
        self.name = name
        self.header = header
        self.content = content
        self.is_spam = is_spam

    def __str__(self):
        return "Document [name={}, header={}, content={}, is_spam={}]".format(self.name, self.header, self.content, self.is_spam)

    def __repr__(self):
        return self.__str__()


class Documents(object):
    def __init__(self):
        pass

    ##
    # Read All Documents
    # @inputs_dir - path to documents
    # @spam_tag   - spam `tag` in name of document
    ##
    @staticmethod
    def get_all_docs(input_dir, spam_tag):
        root = os.path.abspath(os.path.dirname(__file__))
        inputs = os.path.join(root, input_dir)
        docs = []

        paths = list(map(lambda part: os.path.join(inputs, part), os.listdir(inputs)))
        paths.sort()

        for path in paths:
            doc_dir = []

            for doc in os.listdir(path):
                doc_full_path = os.path.join(path, doc)

                with (open(doc_full_path, 'r')) as file:
                    is_spam = (spam_tag in doc_full_path)

                    # -1 to get rid of trailing \n
                    header = file.readline()[:-1]
                    file.readline()
                    content = file.readline()[:-1]

                    # get rid of "header: " prefix and split
                    header = header.split(' ', maxsplit=1)[-1]

                    header = [x for x in header.split(' ') if x]

                    content = [x for x in content.split(' ') if x]

                    doc_dir.append(Document(doc, header, content, is_spam))

            docs.append(doc_dir)

        return docs
