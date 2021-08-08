ALL_FIELDS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
              'marital-status', 'occupation', 'relationship', 'race',
              'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'country']
CONT_FIELDS = ['age', 'fnlwgt', 'education-num',
               'capital-gain', 'capital-loss', 'hours-per-week']
CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))

DATA_TYPES = ['int64', 'object', 'int64', 'object', 'int64', 'object', 'object',
              'object', 'object', 'object', 'int64', 'int64', 'int64', 'object']

# Hyper-parameters for Experiment
NUM_BIN = 10
BATCH_SIZE = 256
EMBEDDING_SIZE = 5
