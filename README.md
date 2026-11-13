Traffic congestion imposes heavy economic, environmental and health costs on large cities, and road pricing is one of the most studied remedies. The idea is often traced to the economist William Vickrey, considered by some to be the father of congestion pricing. In 1952 he proposed that New York City subway fares be raised at peak times and on the busiest sections and lowered elsewhere, and he later made a similar proposal for roads (Vickrey, 1963). Vickrey received the Nobel Memorial Prize in Economic Sciences in 1996. Road pricing later took the form of Electronic Road Pricing (ERP), pioneered in Singapore in 1998. ERP charges tolls per entry based on vehicle size, route taken and time of day (Phang & Toh, 2004). After ERP was introduced, weekday traffic entering Singapore’s restricted zone fell by 24% (U.S. Department of Transportation, 2010). London’s 2003 charge represented public and political recognition of congestion as a distorting externality and of road pricing as an appropriate response (Leape, 2006). Stockholm’s 2006 trial cut cordon traffic by about 22% (Eliasson et al., 2009), and the reduction increased slightly over five years once external factors were controlled for (Börjesson et al., 2012). The Stockholm tax also reduced ambient air pollution by 5–15% and lowered the rate of acute asthma attacks among young children (Simeonova et al., 2021). New York, which began charging in January 2025, reported an 11% reduction in traffic entering its zone in the first year (Governor of New York State, 2026)

Because charges depend on vehicle type, ERP needs reliable vehicle detection and classification. YOLO frames detection as a single neural network that predicts bounding boxes and class probabilities directly from full images in one evaluation (Redmon et al., 2016). YOLO26 adds end-to-end NMS-free inference and a simplified export pipeline (Sapkota et al., 2025). Computer vision is already used in this field. A YOLOv11-based toll system with one camera per plaza reported a mean average precision of 0.895 (Sivakoti, 2024), and researchers have studied New York’s congestion pricing through automated analysis of traffic-camera data (Turkcan et al., 2026). This project applies YOLO26s to detect and classify cars, vans and buses and to compute type-based ERP charges.


**References (APA 7th)**

**Peer-reviewed**



Börjesson, M., Eliasson, J., Hugosson, M. B., & Brundell-Freij, K. (2012). The Stockholm congestion charges—5 years on. Effects, acceptability and lessons learnt. Transport Policy, 20, 1–12.

Eliasson, J., Hultkrantz, L., Nerhagen, L., & Rosqvist, L. S. (2009). The Stockholm congestion-charging trial 2006: Overview of effects. Transportation Research Part A: Policy and Practice, 43(3), 240–250.

Leape, J. (2006). The London congestion charge. Journal of Economic Perspectives, 20(4), 157–176. https://doi.org/10.1257/jep.20.4.157

Phang, S.-Y., & Toh, R. S. (2004). Road congestion pricing in Singapore: 1975 to 2003. Transportation Journal, 43(2), 16–25.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 779–788). https://doi.org/10.1109/CVPR.2016.91

Simeonova, E., Currie, J., Nilsson, P., & Walker, R. (2021). Congestion pricing, air pollution, and children’s health. Journal of Human Resources, 56(4), 971–996. https://doi.org/10.3368/jhr.56.4.0218-9363R2



**Preprints (not peer-reviewed)**



Sapkota, R., Cheppally, R. H., Sharda, A., & Karkee, M. (2025). YOLO26: Key architectural enhancements and performance benchmarking for real-time object detection. arXiv:2509.25164. https://doi.org/10.48550/arXiv.2509.25164

Sivakoti, K. (2024). Vehicle detection and classification for toll collection using YOLOv11 and ensemble OCR. arXiv:2412.12191.

Turkcan, M. K., Tavori, J., Ghaderi, J., Zussman, G., Kostic, Z., & Smyth, A. (2026). A vision-based analysis of congestion pricing in New York City (Technical report). arXiv:2602.03015.



**Official sources**



Governor of New York State. (2026, January 5). Less traffic, better transit: On its first anniversary, Governor Hochul celebrates transformational success of congestion pricing [Press release]. https://www.governor.ny.gov/news/less-traffic-better-transit-its-first-anniversary-governor-hochul-celebrates-transformational

U.S. Department of Transportation, ITS Knowledge Resources. (2010). Lessons learned from international experience in congestion pricing: Final report (Benefit record 2010-B00620). https://www.itskrs.its.dot.gov/2010-b00620

Columbia University Record. (1997, October 10). Vickrey one year later: Catching up with a thinker way ahead of his time. https://econ.columbia.edu/william-vickrey-1914-1996
Vickrey, W. S. (1963). Pricing in urban and suburban transport. The American Economic Review, 53(2), 452–465.
