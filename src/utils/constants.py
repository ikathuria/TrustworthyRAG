# NEO4J
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "test1234"

# Spacy
SPACY_MODEL = "en_core_web_sm"

# Ollama
OLLAMA_URI = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Extractor Configurations
RELATION_PATTERNS = {
    'CREATED_BY': {
        'patterns': [
            'is the author of',
            'authored',
            'created by',
            'developed by',
            'responsible for development',
            'author of'
        ],
        'entity_pairs': [
            ('MALWARE', 'PERSON'),
            ('MALWARE', 'ORGANIZATION')
        ],
        'bidirectional': True
    },
    'EXPLOITS': {
        'patterns': [
            'exploits',
            'targets',
            'attacks',
            'compromises',
            'abuses',
            'exploit',
            'targeting'
        ],
        'entity_pairs': [
            ('MALWARE', 'CVE'),
            ('MALWARE', 'VULNERABILITY'),
            ('MALWARE', 'SYSTEM'),
            ('MALWARE', 'MS_VULN'),
            ('ATTACK_PATTERN', 'SYSTEM'),
            ('ORGANIZATION', 'SYSTEM')
        ],
        'bidirectional': False
    },
    'USES': {
        'patterns': [
            'uses',
            'utilizes',
            'employs',
            'leverages',
            'deploys',
            'use'
        ],
        'entity_pairs': [
            ('MALWARE', 'SYSTEM'),
            ('ORGANIZATION', 'MALWARE'),
            ('PERSON', 'MALWARE'),
            ('ATTACK_PATTERN', 'MALWARE')
        ],
        'bidirectional': False
    },
    'AFFECTS': {
        'patterns': [
            'affects',
            'impacts',
            'influences',
            'damages',
            'affect'
        ],
        'entity_pairs': [
            ('CVE', 'SYSTEM'),
            ('VULNERABILITY', 'SYSTEM'),
            ('MALWARE', 'SYSTEM'),
            ('MS_VULN', 'SYSTEM')
        ],
        'bidirectional': False
    },
    'MEMBER_OF': {
        'patterns': [
            'member of',
            'part of',
            'belongs to',
            'affiliated with',
            'member'
        ],
        'entity_pairs': [
            ('PERSON', 'ORGANIZATION')
        ],
        'bidirectional': False
    },
    'OCCURRED_ON': {
        'patterns': [
            'on',
            'began on',
            'started on',
            'commenced on',
            'took place',
            'occurred'
        ],
        'entity_pairs': [
            ('MALWARE', 'DATE'),
            ('ATTACK_PATTERN', 'DATE'),
            ('CVE', 'DATE')
        ],
        'bidirectional': False
    }
}
