import csv
import sys


def iter_src_tgts(comments):
    prev_prev_line = None
    prev_line = None
    current_line = None

    for _, message in comments:
        prev_prev_line = prev_line
        prev_line = current_line
        current_line = message

        if prev_line:
            yield(None, prev_line, current_line)

        if prev_prev_line:
            yield(prev_prev_line, prev_line, current_line)


docs = {}
if __name__ == '__main__':
    posts_filename = sys.argv[1]
    comments_filename = sys.argv[2]

    with open(posts_filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if len(row['message']) >= len(row['description']):
                text = row['message']
            else:
                text = row['description']
            if text.strip().startswith('http://'):
                continue
            docs[row['post_id']] = {
                'description': row['description'],
                'post_id': row['post_id'],
                'message': row['message'],
                'text': text,
                'comments': []
            }

    with open(comments_filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comment = [row['from_id'], row['message']]
            if row['post_id'] in docs:
                docs[row['post_id']]['comments'].append(comment)

    cntr = 0
    only_comments_f = open('only_comments.txt', 'w')
    for doc_id, doc_data in docs.items():
        if len(doc_data['comments']) > 1:
            for prev_prev_line, prev_line, current_line in iter_src_tgts(doc_data['comments']):
                if prev_prev_line:
                    continue
                    context = "{} _EOS_ {}".format(prev_prev_line, prev_line)
                else:
                    context = prev_line
                    prev_line = prev_line.replace('\r\n', "\n").replace("\n", " ")
                    print(prev_line, file=only_comments_f)
                context = context.replace('\r\n', "\n").replace("\n", " ")
                post = doc_data['text'].replace('\r\n', "\n").replace("\n", " ")
                current_line = current_line.replace('\r\n', "\n").replace("\n", " ")
                src = "{} _EOP_ {}".format(post, context)
                print("{}\t{}".format(src, current_line))
    only_comments_f.close()
