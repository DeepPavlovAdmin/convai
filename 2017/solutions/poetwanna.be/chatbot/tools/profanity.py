"""Surprisingly amusing module for foul language detection.

`badwords` is a pivotal set of this file.
  - if an expression can be split into subwords, it should be split
    (e.g., 'dumb ass' is better than 'dumbass').
"""
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()


def is_profane(tokens, approx_subwords=False):
    """Checks a tokenized sentence for foul language.

    Args:
    -----
    tokens: list of tokens, no need for lowercase
    approx_subwords: bool, when True, every bad word will be matched
        when is a subword (e.g., 'jew' will match with 'jewel').
        It only happens for words close in length up to 2 letter,
        so 'jew' will not match 'jewelery'.
    """
    def iter_pairs(l1, l2):
        for i in range(len(l1) - 1):
            yield l1[i] + l1[i+1]
            yield l2[i] + l2[i+1]

    lower = [t.lower() for t in tokens]
    stemmed = [porter_stemmer.stem(t) for t in lower]
    for t1, t2 in zip(lower, stemmed):
        if t1 in badwords or (t1 != t2 and t2 in badwords):
            return True

    for pair in iter_pairs(lower, stemmed):
        if pair in badwords:
            return True

    if approx_subwords:
        for t in lower + stemmed:
            for bw in badwords:
                if abs(len(t) - len(bw)) <= 2 and bw in t:
                    return True
    return False


excluded_words = {
}

badwords = {
    '2g1c',
    '2 girls 1 cup',
    'acrotomophilia',
    'anal',
    'anilingus',
    'anus',
    'arse',
    'arse hole',
    'ass',
    'ass bag',
    'ass bandit',
    'ass banger',
    'ass bite',
    'ass clown',
    'ass cock',
    'ass cracker',
    'asses',
    'ass face',
    'ass fuck',
    'ass fucker',
    'ass goblin',
    'ass hat',
    'ass head',
    'ass hole',
    'ass hopper',
    'ass jabber',
    'ass jacker',
    'ass lick',
    'ass licker',
    'ass monkey',
    'ass munch',
    'ass muncher',
    'ass nigger',
    'ass pirate',
    'ass shit',
    'ass shole',
    'ass sucker',
    'ass wad',
    'ass wipe',
    'auto erotic',
    'axwound',
    'babeland',
    'baby batter',
    'ball gag',
    'ball gravy',
    'ball kicking',
    'ball licking',
    'ball sack',
    'ball sucking',
    'bampot',
    'bang bros',
    'bare back',
    'barely legal',
    'bare naked',
    'bastard',
    'bastardo',
    'bastinado',
    'bbw',
    'bdsm',
    'beaner',
    'beaver cleaver',
    'beaver lips',
    # 'bestiality',
    'bi curious',
    'big black',
    'big breasts',
    'big knockers',
    'big tits',
    'bimbos',
    'birdlock',
    'bitch',
    'bitch ass',
    'bitches',
    'bitch tits',
    'bitchy',
    'black cock',
    'blonde action',
    'blonde on blonde action',
    'blow j',
    'blow job',
    'blow your l',
    'blue waffle',
    'blumpkin',
    'bollocks',
    'bollox',
    'bondage',
    'boner',
    'boob',
    'boobs',
    'booty call',
    'brother fucker',
    'brown showers',
    'brunette action',
    'bukkake',
    'bull dyke',
    'bullet vibe',
    'bull shit',
    'bumble fuck',
    'bung hole',
    'busty',
    'butt',
    'butt cheeks',
    'butt fucka',
    'butt fucker',
    'butt hole',
    'butt pirate',
    'butt plug',
    'camel toe',
    'camgirl',
    'camslut',
    'camwhore',
    'carpet muncher',
    'chesticle',
    'chinc',
    'chink',
    'choad',
    'chocolate rosebuds',
    'chode',
    'circle jerk',
    'cleveland steamer',
    'clit',
    'clit face',
    'clit fuck',
    'clitoris',
    'clover clamps',
    'cluster fuck',
    'cock',
    'cock ass',
    'cock bite',
    'cock burger',
    'cock face',
    'cock fucker',
    'cock head',
    'cock jockey',
    'cock knoker',
    'cock master',
    'cock mongler',
    'cock mongruel',
    'cock monkey',
    'cock muncher',
    'cock nose',
    'cock nugget',
    'cocks',
    'cock shit',
    'cock smith',
    'cock smoke',
    'cock smoker',
    'cock sniffer',
    'cock sucker',
    'cock waffle',
    'coochie',
    'coochy',
    'coon',
    'cooter',
    'coprolagnia',
    'coprophilia',
    'cornhole',
    'cracker',
    'cum',
    'cum bubble',
    'cum dumpster',
    'cum guzzler',
    'cum jockey',
    'cum ming',
    'cum slut',
    'cum tart',
    'cunnie',
    'cunnilingus',
    'cunt',
    'cunt ass',
    'cunt face',
    'cunt hole',
    'cunt licker',
    'cunt rag',
    'cunt slut',
    'dago',
    'darkie',
    'date rape',
    'deep throat',
    'deggo',
    'dick',
    'dick bag',
    'dick beaters',
    'dick face',
    'dick fuck',
    'dick fucker',
    'dick head',
    'dick hole',
    'dick juice',
    'dick milk',
    'dick monger',
    'dicks',
    'dickslap',
    'dick sneeze',
    'dick sucker',
    'dick sucking',
    'dick tickler',
    'dick wad',
    'dick weasel',
    'dick weed',
    'dick wod',
    'dike',
    'dildo',
    'dipshit',
    'dirty pillows',
    'dirty sanchez',
    'doggie style',
    'doggy style',
    'dog style',
    'dolcett',
    'domination',
    'dominatrix',
    'dommes',
    'donkey punch',
    'doochbag',
    'dookie',
    'double dong',
    'double penetration',
    'douche',
    'douche bag',
    'douche fag',
    'douche waffle',
    'dp action',
    'dumass',
    'dumb ass',
    'dumb fuck',
    'dumb shit',
    'dumshit',
    'dyke',
    'eat my ass',
    'ecchi',
    'ejaculation',
    'erotic',
    'erotism',
    'escort',
    'ethical slut',
    'eunuch',
    'fag',
    'fag bag',
    'fag fucker',
    'faggit',
    'faggot',
    'faggot cock',
    'fagtard',
    'fap',
    'fascist',
    'fatass',
    'fecal',
    'felch',
    'fellatio',
    'feltch',
    'female squirting',
    'femdom',
    'figging',
    'fingering',
    'fisting',
    'flamer',
    'foot fetish',
    'foot job',
    'frotting',
    'fuck',
    'fuck ass',
    'fuck bag',
    'fuck boy',
    'fuck brain',
    'fuck butt',
    'fuck butter',
    'fuck buttons',
    'fucked',
    'fucker',
    'fuck ersucker',
    'fuck face',
    'fuck head',
    'fuck hole',
    'fuckin',
    'fucking',
    'fuck nut',
    'fuck nutt',
    'fuck off',
    'fucks',
    'fuck stick',
    'fuck tard',
    'fuck tart',
    'fuck up',
    'fuck wad',
    'fuck wit',
    'fuck witt',
    'fudge packer',
    'futanari',
    'gang bang',
    'gay',
    'gay ass',
    'gay bob',
    'gay do',
    'gay fuck',
    'gay fuckist',
    'gay lord',
    'gay  sex',
    'gay tard',
    'gay wad',
    'genitals',
    'giant cock',
    'girl on',
    'girl on top',
    'girls gone wild',
    'goatcx',
    'goatse',
    'goddamn',
    'goddamnit',
    'gokkun',
    'golden shower',
    'gooch',
    'good poop',
    'goo girl',
    'gook',
    'goregasm',
    'gringo',
    'grope',
    'group sex',
    'g-spot',
    'guido',
    'guro',
    'hand job',
    'hard core',
    'hard on',
    'heeb',
    'hell',
    'hentai',
    'hitler',
    'ho',
    'hoe',
    'homo',
    'homodumbshit',
    'homoerotic',
    'honkey',
    'hooker',
    'hot chick',
    'how to kill',
    'how to murder',
    'huge fat',
    'humping',
    'incest',
    'intercourse',
    'jackass',
    'jack off',
    'jagoff',
    'jail bait',
    'jap',
    'jerkass',
    'jerk off',
    'jew',
    'jigaboo',
    'jiggaboo',
    'jiggerboo',
    'jizz',
    'juggs',
    'jungle bunny',
    'kike',
    'kinbaku',
    'kinkster',
    'kinky',
    'knobbing',
    'kooch',
    'kootch',
    'kraut',
    'kunt',
    'kyke',
    'lameass',
    'lardass',
    'leather restraint',
    'lemon party',
    'lesbian',
    'lesbo',
    'lezzie',
    'lolita',
    'lovemaking',
    'make me come',
    'male squirting',
    'masturbate',
    'mcfagget',
    'menage a trois',
    'mick',
    'milf',
    'minge',
    'missionary position',
    'mothafucka',
    'mothafuckin\'',
    'mother fucker',
    'mother fucking',
    'mound of venus',
    'mr hands',
    'muff',
    'muff diver',
    'muff diving',
    'munging',
    'nambla',
    'nawashi',
    'nazi',
    'negro',
    'neonazi',
    'nigaboo',
    'nigga',
    'nigger',
    'niggers',
    'niglet',
    'nig nog',
    'nimphomania',
    'nipple',
    'nipples',
    'nsfw',
    'nsfw images',
    'nude',
    'nudity',
    'nut sack',
    'nutsack',
    'nympho',
    'nymphomania',
    'octopussy',
    'omorashi',
    'one cup two girls',
    'one guy one jar',
    'orgasm',
    'orgy',
    'paedophile',
    'paki',
    'panooch',
    'panties',
    'panty',
    'pecker',
    'peckerhead',
    'pedobear',
    'pedophile',
    'pegging',
    'penis',
    'penis banger',
    'penis fucker',
    'penis puffer',
    'phone sex',
    'piece of shit',
    'piss',
    'pissed',
    'pissed off',
    'pissflaps',
    'pissing',
    'piss pig',
    'playboy',
    'pleasure chest',
    'pole smoker',
    'pollock',
    'ponyplay',
    'poof',
    'poon',
    'poonani',
    'poonany',
    'poontang',
    'poop chute',
    'porch monkey',
    'porn',
    'porno',
    'pornography',
    'prick',
    'prince albert piercing',
    'pthc',
    'pubes',
    'punanny',
    'punta',
    'pussies',
    'pussy',
    'pussy licking',
    'puto',
    'queaf',
    'queef',
    'queer',
    'queer bait',
    'queer hole',
    'raghead',
    'raging boner',
    'rape',
    'raping',
    'rapist',
    'rectum',
    'renob',
    'reverse cowgirl',
    'rimjob',
    'rimming',
    'rosy palm',
    'rosy palm and her 5 sisters',
    'ruski',
    'rusty trombone',
    'sadism',
    'sand nigger',
    'scat',
    'schlong',
    'scissoring',
    'scrote',
    'semen',
    'sex',
    'sexo',
    'sexy',
    'shaved beaver',
    'shaved pussy',
    'shemale',
    'shibari',
    'shit',
    'shit ass',
    'shit bag',
    'shit bagger',
    'shit brains',
    'shit breath',
    'shit canned',
    'shit cunt',
    'shit dick',
    'shit face',
    'shit faced',
    'shit head',
    'shit hole',
    'shit house',
    'shit spitter',
    'shit stain',
    'shitter',
    'shittiest',
    'shitting',
    'shitty',
    'shiz',
    'shiznit',
    'shota',
    'shrimping',
    'skank',
    'skeet',
    'skullfuck',
    'slanteye',
    'slut',
    'slutbag',
    's&m',
    'smeg',
    'smut',
    'snatch',
    'snowballing',
    'sodomize',
    'sodomy',
    'spic',
    'spick',
    'splooge',
    'spooge',
    'spook',
    'spread legs',
    'strap on',
    'strappado',
    'strip club',
    'style doggy',
    'suck',
    'suck ass',
    'sucks',
    'suicide girls',
    'sultry women',
    'swastika',
    'swinger',
    'tainted love',
    'tard',
    'taste my',
    'tea bagging',
    'testicle',
    'threesome',
    'throating',
    'thundercunt',
    'tied up',
    'tight white',
    'tit',
    'tit fuck',
    'tits',
    'titties',
    'titty',
    'titty fuck',
    'tongue in a',
    'topless',
    'tosser',
    'towel head',
    'tranny',
    'tribadism',
    'tub girl',
    'tushy',
    'twat',
    'twat lips',
    'twats',
    'twat waffle',
    'twink',
    'twinkie',
    'two girls one cup',
    'unclefucker',
    'undressing',
    'up skirt',
    'urethra play',
    'urophilia',
    'vag',
    'vagina',
    'vajayjay',
    'va-j-j',
    'venus mound',
    'vibrator',
    'violet blue',
    'violet wand',
    'vjayjay',
    'vorarephilia',
    'voyeur',
    'vulva',
    'wank',
    'wank job',
    'wet back',
    'wet dream',
    'white power',
    'whore',
    'whore bag',
    'whore face',
    'women rapping',
    'wop',
    'wrapping men',
    'wrinkled starfish',
    'xx',
    'xxx',
    'yaoi',
    'yellow showers',
    'yiffy',
    'zoophilia',
}

more_badwords = """
    a55
    anal
    anus
    ar5e
    arrse
    arse
    ass
    ass-fucker
    asses
    assfucker
    assfukka
    asshole
    assholes
    asswhole
    a_s_s
    b!tch
    b00bs
    b17ch
    b1tch
    ballbag
    balls
    ballsack
    bastard
    beastial
    beastiality
    bellend
    bestial
    bestiality
    bi\+ch
    biatch
    bitch
    bitcher
    bitchers
    bitches
    bitchin
    bitching
    bloody
    blow job
    blowjob
    blowjobs
    boiolas
    bollock
    bollok
    boner
    boob
    boobs
    booobs
    boooobs
    booooobs
    booooooobs
    breasts
    buceta
    bugger
    bum
    bunny fucker
    butt
    butthole
    buttmuch
    buttplug
    c0ck
    c0cksucker
    carpet muncher
    cawk
    chink
    cipa
    cl1t
    clit
    clitoris
    clits
    cnut
    cock
    cock-sucker
    cock sucker
    cockface
    cockhead
    cockmunch
    cockmuncher
    cocks
    cocksuck
    cocksucked
    cocksucker
    cocksucking
    cocksucks
    cocksuka
    cocksukka
    cok
    cokmuncher
    coksucka
    coon
    cox
    crap
    cum
    cummer
    cumming
    cums
    cumshot
    cunilingus
    cunillingus
    cunnilingus
    cunt
    cuntlick
    cuntlicker
    cuntlicking
    cunts
    cyalis
    cyberfuc
    cyberfuck
    cyberfucked
    cyberfucker
    cyberfuckers
    cyberfucking
    d1ck
    damn
    dick
    dickhead
    dildo
    dildos
    dink
    dinks
    dirsa
    dlck
    dog-fucker
    doggin
    dogging
    donkeyribber
    doosh
    duche
    dyke
    ejaculate
    ejaculated
    ejaculates
    ejaculating
    ejaculatings
    ejaculation
    ejakulate
    f u c k
    f u c k e r
    f4nny
    fag
    fagging
    faggitt
    faggot
    faggs
    fagot
    fagots
    fags
    fanny
    fannyflaps
    fannyfucker
    fanyy
    fatass
    fcuk
    fcuker
    fcuking
    feck
    fecker
    felching
    fellate
    fellatio
    fingerfuck
    fingerfucked
    fingerfucker
    fingerfuckers
    fingerfucking
    fingerfucks
    fistfuck
    fistfucked
    fistfucker
    fistfuckers
    fistfucking
    fistfuckings
    fistfucks
    flange
    fook
    fooker
    fuck
    fucka
    fucked
    fucker
    fuckers
    fuckhead
    fuckheads
    fuckin
    fucking
    fuckings
    fuckingshitmotherfucker
    fuckme
    fucks
    fuckwhit
    fuckwit
    fudge packer
    fudgepacker
    fuk
    fuker
    fukker
    fukkin
    fuks
    fukwhit
    fukwit
    fux
    fux0r
    f_u_c_k
    gangbang
    gangbanged
    gangbangs
    gaylord
    gaysex
    goatse
    God
    god-dam
    god-damned
    goddamn
    goddamned
    hardcoresex
    hell
    heshe
    hoar
    hoare
    hoer
    homo
    homosexual
    hore
    horniest
    horny
    hotsex
    jack-off
    jackoff
    jap
    jerk-off
    jism
    jiz
    jizm
    jizz
    kawk
    knob
    knobead
    knobed
    knobend
    knobhead
    knobjocky
    knobjokey
    kock
    kondum
    kondums
    kum
    kummer
    kumming
    kums
    kunilingus
    l3i\+ch
    l3itch
    labia
    lmfao
    lust
    lusting
    m0f0
    m0fo
    m45terbate
    ma5terb8
    ma5terbate
    masochist
    master-bate
    masterb8
    masterbat
    masterbat3
    masterbate
    masterbation
    masterbations
    masturbate
    mo-fo
    mof0
    mofo
    mothafuck
    mothafucka
    mothafuckas
    mothafuckaz
    mothafucked
    mothafucker
    mothafuckers
    mothafuckin
    mothafucking
    mothafuckings
    mothafucks
    motherfuck
    motherfucked
    motherfucker
    motherfuckers
    motherfuckin
    motherfucking
    motherfuckings
    motherfuckka
    motherfucks
    muff
    mutha
    muthafecker
    muthafuckker
    muther
    mutherfucker
    n1gga
    n1gger
    nazi
    nigg3r
    nigg4h
    nigga
    niggah
    niggas
    niggaz
    nigger
    niggers
    nob
    nobhead
    nobjocky
    nobjokey
    numbnuts
    nutsack
    orgasim
    orgasims
    orgasm
    orgasms
    p0rn
    pawn
    pecker
    penis
    penisfucker
    phonesex
    phuck
    phuk
    phuked
    phuking
    phukked
    phukking
    phuks
    phuq
    pigfucker
    pimpis
    piss
    pissed
    pisser
    pissers
    pisses
    pissflaps
    pissin
    pissing
    pissoff
    poop
    porn
    porno
    pornography
    pornos
    prick
    pron
    pube
    pusse
    pussi
    pussies
    pussy
    rectum
    retard
    rimjaw
    rimming
    s.o.b.
    sadist
    schlong
    screwing
    scroat
    scrote
    scrotum
    semen
    sex
    sh!\+
    sh!t
    sh1t
    shag
    shagger
    shaggin
    shagging
    shemale
    shi\+
    shit
    shitdick
    shite
    shited
    shitey
    shitfuck
    shitfull
    shithead
    shiting
    shits
    shitted
    shitter
    shitting
    shitty
    skank
    slut
    sluts
    smegma
    smut
    snatch
    son-of-a-bitch
    spac
    spunk
    s_h_i_t
    t1tt1e5
    t1tties
    teets
    teez
    testical
    testicle
    tit
    titfuck
    tits
    titt
    tittie5
    tittiefucker
    titties
    tittyfuck
    tittywank
    titwank
    tosser
    turd
    tw4t
    twat
    twathead
    twatty
    twunt
    twunter
    v14gra
    v1gra
    vagina
    viagra
    vulva
    w00se
    wang
    wank
    wanker
    wanky
    whoar
    whore
    willies
    willy
    xrated
    xxx
"""

danger = """
    brezhnev
    castro
    che guevara
    christ
    death to
    engels
    gaddafi
    hussein
    lenin
    kill all
    kill every
    khrushchev
    jaruzelski
    jesus
    jong il
    jong-il
    jong un
    jong-un
    marx
    murder all
    murder every
    mussolini
    pol pot
    stalin
    tse tung
    tse-tung
    zedong
"""

chatterbot_bad = {
    'agent ruby',
    'marijuana',
    'cigarettes'
}

more_badwords = [x.strip() for x in more_badwords.split('\n')]
badwords.update(more_badwords)

danger = [x.strip() for x in danger.split('\n')]
badwords.update(danger)

badwords.update(chatterbot_bad)

# ass face --> assface
badwords.update([u''.join(bw.split()) for bw in badwords])
# ass face --> ass-face
badwords.update([u'-'.join(bw.split()) for bw in badwords])

badwords.update(excluded_words)
