import re

import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.MENTION)


def camel_case_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


def clean(text):
    text = p.clean(text)
    text = re.sub(r'^RT ', '', text)
    text = ' '.join(camel_case_split(text))
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r"\d+", "number", text)
    if len(text.strip().split()) < 3:
        return None
    return text.lower().strip()
