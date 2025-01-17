Hardware accelerated orbits and light curves of exoplanet systems I.


Light curve observations of exoplanets are the primary way of understanding their properties and dynamics. 
Dan, So and I worked on 3 different things. The first thing is a re-implementation of exoplanet in JAX, called jaxoplanet. The second is a re-implementation of starry in JAX. And finally, the last is an implementation of a pixelated star in JAX.
The goal of these three implementations is to model the orbits and light curves of exoplanet system. This also include radial velocities and other observable we get from these systems. The key similarity of these three implmentation is that host stars are not uniform and rotate, which leads to an extra signal on top of the planet's signals.

So now I have to write a paper about that. I already wrote a paper about the two first aspects but Dan think (and he might be right) that depending on the motivation this might be a bit incomplete.

Light curve observations of exoplanets are the primary way of understanding their properties and dynamics. 

The study of exoplanet systems has been revolutionized by high-precision photometric surveys, which monitor the brightness variations of distant stars. A key observational technique, transit photometry, captures periodic dimming in a star’s light curve as an orbiting exoplanet passes in front of it. These light curves encode critical information about planetary characteristics, such as radius, orbital inclination, and even atmospheric composition through transit spectroscopy.

Exoplanet host stars are not uniform light sources; they exhibit surface inhomogeneities such as limb darkening, starspots, and granulation. These effects significantly distort observed light curves, introducing degeneracies in parameter estimation if not properly accounted for. Limb darkening, for instance, causes the stellar disk to appear dimmer toward its edges, altering the transit depth and shape. Starspots and faculae can further introduce wavelength- and time-dependent variations, complicating the disentanglement of planetary signals from stellar activity. Accurately modeling light curves thus requires realistic stellar surface representations and numerical techniques capable of handling such complexities.

Generating such light curves forward models involves computing the flux blocked by the planet as it transits a spatially varying stellar disk, integrated over the visible surface at each time step. This is computationally demanding, particularly when accounting for the orbital dynamic of the two-body system. The inverse problem, which consist in inferring system parameters from observed light curves, is even more challenging. It typically involves Bayesian inference methods or machine learning techniques that require repeated evaluation of the light curve forward model to explore parameter space efficiently.

Finally, the rapid expansion of exoplanet observations, driven by next-generation observatories like the James Webb Space Telescope (JWST), or large-scale surveys like TESS, increase the precision and volume of data that need to be analyzed. In addition, combining multi-instrument datasets - each with unique systematics and spectral coverage - further complicates modeling efforts, significantly increasing computational costs and limiting the feasibility of traditional inference techniques. 

To keep pace with the growing complexity of data and stellar surface representations, forward models must be designed for efficiency and could benefit from hardware accelerators like Graphics Processing Units (GPUs). While not yet widely adopted for these tasks, GPUs offer substantial potential for accelerating Bayesian inference and simulation techniques, paving the way for more scalable exoplanet data analysis.

In this work, we present hardware-accelerated approaches to computing exoplanet orbits and light curves, leveraging GPU-based parallelism to improve efficiency.

# Hardware accelerated light curves of exoplanet systems 

## Introduction

Light curves are an essential observable to infer exoplanets parameters. By comparing these data to precise forward models, one can retrieve the physical and dynamic properties of these distant world only based on the light received from their host stars. With the lunch of next-generation observatories like the James Webb Space Telescope (JWST), or large-scale surveys like TESS, the volume and precision of these datasets have drastically increased. This lead to the modeling of higher order effect, such as transit time variation or stellar spot occultation, and the combination of data from different instruments.

But models and data complexity increases:
- Modeling more precise and voluminous data
- Modeling more complex stellar surfaces
- Modeling different instruments together

Stack of tools is also getting complex:
- Forward models for different applications
- Bayesian inference tools
- Differentiation, hardware accelerators
- Deprecation

JAX offers a good solution:
- Ecosystem of tools
- Differentiation
- Hardware acceleration

We present 3 JAX forward models of light curves serving 3 different regimes:
- Transits of limb-darkened stars 
- Transits of low-order non-uniform stars (eclipse mapping)
- Transits of high-order non-uniform stars (spot occultation)

In this paper we present the first two that are based on spherical harmonics to describe the stellar surface.

## Light curve models
## Implementation details
## Performance
## Conclusion

The study of exoplanet systems has been revolutionized by high-precision photometric surveys, which monitor the brightness variations of distant stars. A key observational technique, transit photometry, captures periodic dimming in a star’s light curve as an orbiting exoplanet passes in front of it. These light curves encode critical information about planetary characteristics, such as radius, orbital inclination, and even atmospheric composition through transit spectroscopy.


Light curves are an essential observable to infer exoplanet parameters, providing insights into their orbit, their size and the properties of their atmospheres. Exoplanet host stars are not uniform light sources; they exhibit surface inhomogeneities such as limb darkening, starspots, and granulation. These effects significantly distort observed light curves, introducing degeneracies in parameter estimation if not properly accounted for. Limb darkening, for instance, causes the stellar disk to appear dimmer toward its edges, altering the transit depth and shape. Starspots and faculae can further introduce wavelength- and time-dependent variations, complicating the disentanglement of planetary signals from stellar activity. Accurately modeling light curves thus requires realistic stellar surface representations and numerical techniques capable of handling such complexities.

Light curve observations of exoplanets encode critical information about their properties, such as radius, orbital configuration, and atmospheric composition through transit spectroscopy. Exoplanet host stars are not uniform light sources; they exhibit surface inhomogeneities such as limb darkening, starspots, and granulation. These effects significantly distort observed light curves, introducing degeneracies in parameter estimation if not properly accounted for. Limb darkening, for instance, causes the stellar disk to appear dimmer toward its edges, altering the transit depth and shape. Starspots and faculae can further introduce wavelength- and time-dependent variations, complicating the disentanglement of planetary signals from stellar activity. Accurately modeling light curves thus requires realistic stellar surface representations and numerical techniques capable of handling such complexities.

This complexity is enhance by the increased volume and precision of

However, exoplanet host stars are not uniform light sources; they exhibit surface inhomogeneities such as limb darkening, starspots, and granulation. These effects significantly distort observed light curves, introducing degeneracies in parameter estimation if not properly accounted for. Limb darkening, for instance, causes the stellar disk to appear dimmer toward its edges, altering the transit depth and shape. Starspots and faculae can further introduce wavelength- and time-dependent variations, complicating the disentanglement of planetary signals from stellar activity. Accurately modeling light curves thus requires realistic stellar surface representations and numerical techniques capable of handling such complexities.

In recent years, the expansion of exoplanet observations, driven by next-generation observatories like the James Webb Space Telescope (JWST), or large-scale surveys like TESS, increase the volume of data that need to be analyzed. In addition, as these data become more precise, there is a need to model higher-order effects, such as modeling the occultation of starspots by planetary companions. Given the information that we try to retrieve, data analysis are made increasingly more complex by combining data from different epochs and instruments.

The launch of next-generation observatories like the James Webb Space Telescope (JWST), or large-scale surveys like TESS has led to an increased volume of data. In addition, as these data become more precise, the community starts to focus more and more on higher-order effects, such as spot occultation, which enhance models complexity. Given the information that we try to retrieve, data analysis are made increasingly more complex by combining data from different epochs and instruments.


But models and data complexity increases:
- Modeling more precise and voluminous data
- Modeling more complex stellar surfaces
- Modeling different instruments together

Stack of tools is also getting complex:
- Forward models for different applications
- Bayesian inference tools
- Differentiation, hardware accelerators
- Deprecation

JAX offers a good solution:
- Ecosystem of tools
- Differentiation
- Hardware acceleration


In this paper we present the first two that are based on spherical harmonics to describe the stellar surface.





Light curves serve as a fundamental observable for inferring exoplanet parameters.


 offering insights into their orbital configuration and atmospheres. As the field advances, both the complexity of models and the volume of observational data continue to grow. High-precision instruments such as JWST and TESS provide an unprecedented influx of data, necessitating more sophisticated models capable of capturing non-uniform surface features of exoplanets host stars and performing multi-instrument analyses.

Simultaneously, the computational demands of exoplanet characterization have intensified. The modeling pipeline involves forward models for light curve synthesis, Bayesian inference techniques for parameter estimation, and differentiation methods for optimization. The integration of hardware accelerators, such as GPUs and TPUs, has become crucial to managing these computational challenges efficiently.

In this work, we explore hardware-accelerated approaches for computing exoplanet orbits and light curves, optimizing both performance and accuracy. By leveraging modern computational tools, we aim to streamline the inference process and enhance the scalability of exoplanet studies.



-------


Light curve observations of transiting exoplanets offer insights into their dynamics and physical characteristics. In order to infer these properties, light curves have to be compared with forward models. Generating these models involves computing the flux blocked by the planet as it transits a spatially varying stellar disk, integrated over the visible surface at each time step. This is computationally demanding, particularly when accounting for the orbital dynamic of the two-body system. The inverse problem, which consist in inferring system parameters from observed light curves, is even more challenging. It typically involves Bayesian inference methods or machine learning techniques that require repeated evaluation of the light curve forward model to explore parameter space efficiently. 

-------

Light curves are an essential observable to infer exoplanets parameters. By comparing these data to precise forward models, one can retrieve the physical and dynamic properties of these distant world only based on the light received from their host stars. With the lunch of next-generation observatories like the James Webb Space Telescope (JWST), or large-scale surveys like TESS, the volume and precision of these datasets have drastically increased. This lead to the modeling of higher order effect, such as transit time variation or stellar spot occultation, and the combination of data from different instruments.


Light curve provide a geat tool to infer the properties of exoplanets, from their orbital dynamic to their physical characteristics like size and atmospheric composition. To do that, one has to compare an exoplanet light curve (let say a transt light curve) with a precise forward model that is capableo f modeling the non-uniform surface of a star. Recently, their have been a increase in datasets, both in volume and in precixiosn, which lead the community to do two different things. One is to want to model higher order effect observed in light curves like transit time variations on very long time scale or small spot occultations. the second is to combine datasets from different instruments, accros different wavelenght bands, and perform inferrence on a very large number of parameter. This lead to a very complex stack of tools, some of them being progressively deprecated. In addition, in order to make inference fatser, one may want differentiable models so they can usegradient-based inferrence. GPUs qre being more qnd more used in other field and coud be used for this application too. So their is an interest in having forward models on hardware accelerators with differentiability and compatible with modern tools to perforn probabilistic inferrence.


-------

Light curves are a powerful tool for inferring exoplanet properties, from orbital dynamics to physical characteristics such as their size and atmospheric composition. To extract meaningful insights, observed light curves—such as transit light curves—must be compared to precise forward models that account for stellar surface inhomogeneities. The increasing volume and precision of observational datasets have introduced new challenges and opportunities. On one hand, higher-order effects, such as long-term transit time variations and small-scale spot occultations, are becoming detectable, necessitating more sophisticated modeling. On the other hand, the need to combine multi-instrument, multi-wavelength datasets has expanded inference to a larger parameter space, making computational pipelines increasingly complex. At the same time, many existing tools are becoming outdated and unable to fully leverage modern data and computational techniques. To keep pace, efficient and scalable modeling approaches are required. Differentiable models, which enable gradient-based inference, offer a promising solution, particularly when paired with hardware accelerators such as GPUs. As GPUs gain traction in scientific computing, they present an opportunity to enhance forward modeling in exoplanet studies. Consequently, developing differentiable, hardware-accelerated forward models compatible with modern probabilistic inference frameworks is a crucial step toward handling the next generation of exoplanet observations.

