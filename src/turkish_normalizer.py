# get zemberek from https://github.com/ahmetaa/zemberek-nlp

from os.path import join

from jpype import JClass, JString, getDefaultJVMPath, startJVM

ZEMBEREK_PATH: str = join('zemberek', 'bin', 'zemberek-full.jar')

startJVM(
    getDefaultJVMPath(),
    '-ea',
    f'-Djava.class.path={ZEMBEREK_PATH}',
    convertStrings=False
)

TurkishMorphology: JClass = JClass('zemberek.morphology.TurkishMorphology')
TurkishSentenceNormalizer: JClass = JClass(
    'zemberek.normalization.TurkishSentenceNormalizer'
)
Paths: JClass = JClass('java.nio.file.Paths')

normalizer = TurkishSentenceNormalizer(
    TurkishMorphology.createWithDefaults(),
    Paths.get(
        join('zemberek', 'data', 'normalization')
    ),
    Paths.get(
        join('zemberek', 'data', 'lm', 'lm.2gram.slm')
    )
)


def normalize(text):
    return str(normalizer.normalize(JString(text)))


def normalize_df(df, text_col):
    df[text_col] = df[text_col].apply(normalize)
    return df
