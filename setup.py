from setuptools import setup, find_namespace_packages

setup(
    name='sacnet',
    packages=find_namespace_packages(include=["sacnet", "sacnet.*"]),
    version='1.0.0',
    description='SACNet: A Multiscale Diffeomorphic Convolutional Registration Network with Prior Neuroanatomical Constraints for Flexible Susceptibility Artifact Correction in Echo Planar Imaging.',
    url='https://github.com/RicardoZiTseng/SACNet',
    author='Zilong Zeng',
    author_email='zilongzeng@mail.bnu.edu.cn',
    license='GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007',
    install_requires=[],
    entry_points={
        'console_scripts': [
            'sacnet_train = sacnet.run.train:main',
            'sacnet_predict_multipe = sacnet.run.predict_multipe:main',
            'sacnet_predict_singlepe = sacnet.run.predict_singlepe:main',
            'sacnet_rotate_bvecs = sacnet.run.rotate_bvecs:main',
            'sacnet_apply_fieldmap = sacnet.run.apply_fieldmap:main',
            'sacnet_show_avaliable_model_info = sacnet.run.display_pretrained_model:print_available_pretrained_models',
            'sacnet_print_pretrained_model_info = sacnet.run.display_pretrained_model:print_pretrained_model_info',
        ]
    },
    # python_requires="==3.6.*",
    # install_requires=[
    #     "art==5.3",
    #     "torch==1.9.1",
    #     "nibabel==3.2.1",
    #     "matplotlib==3.3.4",
    #     "batchgenerators==0.23"
    # ],
    keywords=['susceptibility artifact correction', 'echo planar imaging', 
            'diffeomorphic image registration', 'deep learning']
)
