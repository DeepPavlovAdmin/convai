import requests
import argparse


parser = argparse.ArgumentParser(description='Get answer')
parser.add_argument('--paragraph', type=str, help='Paragraph', required=True)
parser.add_argument('--question', type=str, help='Question', required=True)

if __name__ == '__main__':
    args = parser.parse_args()
    r = requests.get(
        'http://bidaf:1995/submit',
        params={'question': args.question, 'paragraph': args.paragraph})
    print(r.json()['result'])
