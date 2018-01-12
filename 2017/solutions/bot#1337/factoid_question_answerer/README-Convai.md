# Requirements

- Python 3
- Docker ver. 17.03+:

   - Ubuntu: https://docs.docker.com/engine/installation/linux/ubuntu/#install-using-the-repository
   - Mac: https://download.docker.com/mac/stable/Docker.dmg

- pip3 install requests==2.16.5


# Setup

- `./setup`

This script downloads and runs docker with bi-att-flow.

# Usage

`python3 get_answer.py --question <question> --paragraph <paragraph>`

It outputs an answer.

## Example

```
python3 get_answer.py --question "where is the Victoria and Albert Museum located?" \
  --paragraph "The Victoria and Albert Museum (often abbreviated as the V&A), London, is the world's largest museum of decorative arts and design, housing a permanent collection of over 4.5 million objects. It was founded in 1852 and named after Queen Victoria and Prince Albert."
```

Output: `London`




