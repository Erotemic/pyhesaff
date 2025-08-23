"""
Liberate the kwimage demodata grabber so we don't depend on shapely

import liberator
import kwimage
lib = liberator.Liberator()
lib.add_dynamic(kwimage.grab_test_image_fpath)
lib.expand(['kwimage'])
print(lib.current_sourcecode())
"""

import ubelt as ub


def _grabdata_with_mirrors(url, mirror_urls, grabkw):
    fpath = None
    verbose = 1
    try:
        fpath = ub.grabdata(url, **grabkw)
    except Exception as main_ex:
        if verbose:
            print(f'Failed to grab main url: {main_ex}')
            print('Fallback to mirrors:')
        # urllib.error.HTTPError
        for idx, mirror_url in enumerate(mirror_urls):
            try:
                fpath = ub.grabdata(mirror_url, **grabkw)
            except Exception as ex:
                if verbose:
                    print(f'Failed to grab mirror #{idx}: {ex}')
            else:
                break
        if fpath is None:
            raise main_ex
    return fpath


_TEST_IMAGES = {
    'airport': {
        'fname': 'airport.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/9/9e/Beijing_Capital_International_Airport_on_18_February_2018_-_SkySat_%281%29.jpg',
        'note': 'An overhead image the an airport',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb7ea71cc6eae69303aa/download',
        ],
        'ipfs_cids': [
            'bafkreif76x4sclk4o7oup4vybzo4dncat6tycoyi7q43kbbjisl3gsb77q',
        ],
        'sha256': 'bff5f9212d5c77dd47f2b80e5dc1b4409fa7813b08fc39b504294497b3483ffc',
        'sha512': '957695b319d8b8266e4eece845b016fbf2eb4f1b6889d7374d29ab812f752da77e42a6cb664bf15cc38face439bd60070e85e5b7954be15fc354b07b353b9582',
        'properties': {
            'shape': (
                868,
                1156,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'amazon': {
        'fname': 'amazon.jpg',
        'url': 'https://data.kitware.com/api/v1/file/611e9f4b2fa25629b9dc0ca2/download',
        'note': 'An overhead image of the amazon rainforest',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb85a71cc6eae69303ad/download',
        ],
        'ipfs_cids': [
            'bafybeia3telu2s742xco3ap5huh4tk45cikwuxczwhrd6gwc3rcuat7odq',
        ],
        'sha256': 'ef352b60f2577692ab3e9da19d09a49fa9da9937f892afc48094988a17c32dc3',
        'sha512': '80f3f5a5bf5b225c36cbefe44e0c977bf9f3ea53658a97bc2d215405587f40dea6b6c0f04b5934129b4c0265616846562c3f15c9aba61ae1afaacd13c047c9cb',
        'properties': {
            'shape': (
                3000,
                3836,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'astro': {
        'fname': 'astro.png',
        'url': 'https://i.imgur.com/KXhKM72.png',
        'note': 'An image of Eileen Collins.',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb78a71cc6eae69303a7/download',
        ],
        'ipfs_cids': [
            'bafybeif2w42xgi6vkfuuwmn3c6apyetl56fukkj6wnfgzcbsrpocciuv3i',
        ],
        'sha256': '9f2b4671e868fd51451f03809a694006425eee64ad472f7065da04079be60c53',
        'sha512': 'de64fcb37e67d5b5946ee45eb659436b446a9a23ac5aefb6f3cce53e58a682a0828f5e8435cf7bd584358760d59915eb6e37a1b69ca34a78f3d511e6ebdad6fd',
        'properties': {
            'shape': (
                512,
                512,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'carl': {
        'fname': 'carl.jpg',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/b/be/Carl_Sagan_Planetary_Society.JPG',
        'mirrors': [
            'https://i.imgur.com/YnrLyry.jpg',
            'https://data.kitware.com/api/v1/file/647cfb8da71cc6eae69303b0/download',
        ],
        'note': 'An image of Carl Sagan.',
        'ipfs_cids': [
            'bafkreieuyks2z7stz56q63dvk555sr57kwnevgoruiaob7ffg5qcvftnui',
        ],
        'sha256': '94c2a5acfe53cf7d0f6c75577bd947bf559a4a99d1a200e0fca537602a966da2',
        'sha512': 'dc948163225157b85a968b2614cf2a2416b98d8b7b115ce8e046744e64e0f01150e539c06e78fc58306725188ee84f443414abac2e95dc11a8f2435df97ab6d4',
        'properties': {
            'shape': (
                448,
                328,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'lowcontrast': {
        'fname': 'lowcontrast.jpg',
        'url': 'https://i.imgur.com/dyC68Bi.jpg',
        'note': 'A low contrast image of a lobster',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb93a71cc6eae69303b3/download',
        ],
        'ipfs_cids': [
            'bafkreictevzkeroswqavqneizt47am7fsyg4t47vnogknojtvcmg5spjly',
        ],
        'sha256': '532572a245d2b401583488ccf9f033e5960dc9f3f56b8ca6b933a8986ec9e95e',
        'sha512': '68d37c11a005168791e6a6ca018d34c6ee305c76a38fa8c93ccfaf4520f2f01d690b218b4ad6fbac36790104a670a154daa2da14850b5de0cc7c5d6843e5b18a',
        'properties': {
            'shape': (
                267,
                400,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 85,
            'max_value': 193,
        },
    },
    'paraview': {
        'fname': 'paraview.png',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/46/ParaView_splash1.png',
        'note': 'The paraview logo',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb97a71cc6eae69303b6/download',
        ],
        'ipfs_cids': [
            'bafkreiefsqr257hban5sw2kzw5gxwe32ieckzvw3swusi6v3e3bnbkutxa',
        ],
        'sha256': '859423aefce1037b2b6959b74d7b137a4104acd6db95a9247abb26c2d0aa93b8',
        'sha512': '25e92fe7661c0d9caf8eb919f6a9e76ed1bc689b1c599ad0786a47b86578961b07746a8303deb9efdab2bb562c700751d8cf6555e628bb65cb7ea74e8da8ad23',
        'properties': {
            'shape': (
                106,
                462,
                4,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'parrot': {
        'fname': 'parrot.png',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/f/fa/Grayscale_8bits_palette_sample_image.png',
        'note': 'An standard parrot test image',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfb9ca71cc6eae69303b9/download',
        ],
        'ipfs_cids': [
            'bafkreih23vgn3xcg4qyylgmueholdlu5hotnco23nufmybjgr7dsi3z6le',
        ],
        'sha256': 'fadd4cdddc46e43185999421dcb1ae9d3ba6d13b5b6d0acc05268fc7246f3e59',
        'sha512': '542f08ae6228483aa418ed1108d99a63805292bae43388256ea3edad780f7de2654ace72efcea4259b44a41784c364543fe763d4e4c65c90221be4b70e2d056c',
        'properties': {
            'shape': (
                200,
                150,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'stars': {
        'fname': 'stars.png',
        'url': 'https://i.imgur.com/kCi7C1r.png',
        'note': 'An image of stars in the night sky',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfba7a71cc6eae69303bf/download',
        ],
        'ipfs_cids': [
            'bafkreibwhenu2nvuwxrfs7ct7fdfsumravbpx3ec6wqccnyvowor32lrj4',
        ],
        'sha256': '36391b4d36b4b5e2597c53f9465951910542fbec82f5a0213715759d1de9714f',
        'sha512': 'e19e0c0c28c67441700cf272cb6ae20e5cc0baee24e5527e096e61e290ca823913224cdbabb884c5550e73587192428c0650921a00630c82f45c4eddf52c652f',
        'properties': {
            'shape': (
                256,
                256,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'superstar': {
        'fname': 'superstar.jpg',
        'url': 'https://data.kitware.com/api/v1/file/661b2df25165b19d36c87d1c/download',
        'note': 'A pre-rendered superstar',
        'mirrors': [
            'https://i.imgur.com/d2FHuIU.png',
        ],
        'sha256': '90fbba3e0985988f43440b742162535eb6458be9fdbd9dc6db6629f1bd4ded29',
        'sha512': '5471f17234c8e47370ff6782b8b013fce2799e05e8a54739da75cc43adb59bf3d34262024122d893bb0b243f3bbfcc67be00369bd8b0de3aa5328221c62ab419',
        'properties': {
            'shape': (
                64,
                64,
            ),
            'dtype': 'uint8',
            'min_value': 26,
            'max_value': 204,
        },
        'ipfs_cids': [
            'bafkreieq7o5d4cmftchugraloqqweu26wzcyx2p5xwo4nw3gfhy32tpnfe',
        ],
    },
    'pm5644': {
        'fname': 'Philips_Pattern_pm5644.png',
        'url': 'https://upload.wikimedia.org/wikipedia/commons/4/47/Philips_Pattern_pm5644.png',
        'note': 'A test pattern good for checking aliasing effects',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfba2a71cc6eae69303bc/download',
        ],
        'ipfs_cids': [
            'bafkreihluuadifesmsou7jhihnjabk577jthhxs54tba5vtj33pladjzie',
        ],
        'sha256': 'eba500341492649d4fa4e83b5200abbffa6673de5de4c20ed669dedeb00d3941',
        'sha512': '8841ccd59b41dde98385e93531837668f09fafa42cfbdf27bf7c1088028596e3c82da8cad102543b330e1bba97476060ce002864360da76b2b3116647d2a79d8',
        'properties': {
            'shape': (
                576,
                1024,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'tsukuba_l': {
        'fname': 'tsukuba_l.png',
        'url': 'https://i.imgur.com/DhIKgGx.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfbaba71cc6eae69303c2/download',
            'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imL.png',
        ],
        'ipfs_cids': [
            'bafkreihcsfciih2oeiaordvwvwjiz6r64dcvzswaukctfsjjhrff4cziju',
        ],
        'sha256': 'e29144841f4e2200e88eb6ad928cfa3ee0c55ccac0a28532c9293c4a5e0b284d',
        'sha512': '51b8df8fb08f12609676923bb473c76b8ef9d73ce2c5493bca00b7b4b0eec7b298ce33f0bf860cc94c8b7cda8e69e021674e5a7ddaf0a1f007318053e4985740',
        'properties': {
            'shape': (
                288,
                384,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
    'tsukuba_r': {
        'fname': 'tsukuba_r.png',
        'url': 'https://i.imgur.com/38RST9H.png',
        'mirrors': [
            'https://data.kitware.com/api/v1/file/647cfbb0a71cc6eae69303c5/download',
            'https://raw.githubusercontent.com/tohojo/image-processing/master/test-images/middlebury-stereo-pairs/tsukuba/imR.png',
        ],
        'ipfs_cids': [
            'bafkreih3j2frkyobo6u2xirwso6vo3ioa32xpc4nitq4dtc4lxjv2x6r2q',
        ],
        'sha256': 'fb4e8b1561c177a9aba23693bd576d0e06f5778b8d44e1c1cc5c5dd35d5fd1d4',
        'sha512': '04da24efa0037aaad7a72a19d2210dd64f39f1a703d12fd1b379c3d6a9fb8695f33584d566b6159eb9aebce5b9b930b52df4b2ae7e90fcf66014711063635c27',
        'properties': {
            'shape': (
                288,
                384,
                3,
            ),
            'dtype': 'uint8',
            'min_value': 0,
            'max_value': 255,
        },
    },
}


def grab_test_image_fpath(key='astro'):
    """
    Ensures that the test image exists (this might use the network) and returns
    the cached filepath to the requested image.

    Args:
        key (str): which test image to grab. Valid choices are:
            astro - an astronaught
            carl - Carl Sagan
            paraview - ParaView logo
            stars - picture of stars in the sky
            OR can be an existing path to an image

        dsize (None | Tuple[int, int]):
            if specified, we will return a variant of the data with the
            specific dsize

        overviews (None | int):
            if specified, will return a variant of the data with overviews

        allow_fallback (bool):
            if True, and for some reason (e.g. network issue) we cannot grab
            the requested image, generate a random image based with expected
            metadata.

    Returns:
        str: path to the requested image
    """
    try:
        item = _TEST_IMAGES[key]
    except KeyError:
        valid_keys = sorted(_TEST_IMAGES.keys())
        cand = ub.Path(key)
        if cand.exists() and cand.is_file():
            return cand
        else:
            raise KeyError(
                'Unknown key={!r}. Valid keys are {!r}'.format(
                    key, valid_keys))
    if not isinstance(item, dict):
        item = {'url': item}

    grabkw = {
        'appname': 'kwimage/demodata',
    }
    hasher_priority = ['sha256']
    for hasher in hasher_priority:
        if hasher in item:
            grabkw.update({
                'hash_prefix': item[hasher],
                'hasher': hasher,
            })
            break
    if 'fname' in item:
        grabkw['fname'] = item['fname']

    ipfs_gateways = [
        'https://ipfs.io/ipfs',
        'https://dweb.link/ipfs',
        # 'https://gateway.pinata.cloud/ipfs',
    ]
    url = item['url']
    mirror_urls = []
    if 'mirrors' in item:
        mirror_urls += item['mirrors']
    if 'ipfs_cids' in item:
        for cid in item['ipfs_cids']:
            for gateway in ipfs_gateways:
                ipfs_url = gateway + '/' + cid
                mirror_urls.append(ipfs_url)

    fpath = _grabdata_with_mirrors(url, mirror_urls, grabkw)

    return fpath
