## Test seasons 2022/23-2025/26, 6,228 matches with bookmaker odds

| model                     |   log_loss |   brier |    rps |   accuracy |
|:--------------------------|-----------:|--------:|-------:|-----------:|
| Home-win prior (baseline) |     1.0729 |  0.6492 | 0.2302 |     0.4379 |
| Elo only                  |     0.983  |  0.5859 | 0.1994 |     0.5289 |
| Logistic regression       |     0.9806 |  0.5843 | 0.1987 |     0.5316 |
| LightGBM                  |     0.9805 |  0.5838 | 0.1984 |     0.5292 |
| Market average            |     0.9655 |  0.574  | 0.194  |     0.5422 |
| Pinnacle (opening)        |     0.9651 |  0.5738 | 0.1939 |     0.5414 |
| Pinnacle (closing)        |     0.963  |  0.5725 | 0.1933 |     0.5454 |

## All 7,082 test matches (Pinnacle stopped publishing mid-2025/26)

| model                     |   log_loss |   brier |    rps |   accuracy |
|:--------------------------|-----------:|--------:|-------:|-----------:|
| Home-win prior (baseline) |     1.0732 |  0.6494 | 0.2303 |     0.4373 |
| Elo only                  |     0.9871 |  0.5886 | 0.2007 |     0.5246 |
| Logistic regression       |     0.9834 |  0.5862 | 0.1996 |     0.5292 |
| LightGBM                  |     0.9841 |  0.5861 | 0.1995 |     0.527  |
| Market average            |     0.9691 |  0.5763 | 0.1951 |     0.5402 |

## By league

| league         | model                     |   log_loss |   brier |    rps |   accuracy |
|:---------------|:--------------------------|-----------:|--------:|-------:|-----------:|
| Bundesliga     | Home-win prior (baseline) |     1.0729 |  0.6493 | 0.2311 |     0.4367 |
| Bundesliga     | Elo only                  |     0.9878 |  0.5897 | 0.2018 |     0.5211 |
| Bundesliga     | Logistic regression       |     0.9836 |  0.5864 | 0.2002 |     0.523  |
| Bundesliga     | LightGBM                  |     0.9839 |  0.5862 | 0.2001 |     0.5183 |
| Bundesliga     | Market average            |     0.97   |  0.5765 | 0.1956 |     0.5361 |
| Bundesliga     | Pinnacle (opening)        |     0.9694 |  0.5762 | 0.1954 |     0.5351 |
| Bundesliga     | Pinnacle (closing)        |     0.9684 |  0.5752 | 0.1954 |     0.5464 |
| Premier League | Home-win prior (baseline) |     1.0645 |  0.6435 | 0.2317 |     0.4511 |
| Premier League | Elo only                  |     0.9732 |  0.5792 | 0.2002 |     0.5385 |
| Premier League | Logistic regression       |     0.9676 |  0.5754 | 0.1984 |     0.5444 |
| Premier League | LightGBM                  |     0.9713 |  0.5772 | 0.1989 |     0.5348 |
| Premier League | Market average            |     0.9546 |  0.5664 | 0.1943 |     0.5578 |
| Premier League | Pinnacle (opening)        |     0.9547 |  0.5664 | 0.1943 |     0.5563 |
| Premier League | Pinnacle (closing)        |     0.9497 |  0.5626 | 0.1926 |     0.5578 |
| Ligue 1        | Home-win prior (baseline) |     1.0696 |  0.6473 | 0.2338 |     0.4393 |
| Ligue 1        | Elo only                  |     0.9937 |  0.5915 | 0.2065 |     0.5275 |
| Ligue 1        | Logistic regression       |     0.9966 |  0.5938 | 0.2075 |     0.5179 |
| Ligue 1        | LightGBM                  |     0.9968 |  0.5936 | 0.2075 |     0.5179 |
| Ligue 1        | Market average            |     0.9768 |  0.5811 | 0.2018 |     0.5328 |
| Ligue 1        | Pinnacle (opening)        |     0.9765 |  0.581  | 0.2017 |     0.5319 |
| Ligue 1        | Pinnacle (closing)        |     0.9761 |  0.5812 | 0.2014 |     0.5362 |
| Serie A        | Home-win prior (baseline) |     1.0908 |  0.6615 | 0.2288 |     0.4073 |
| Serie A        | Elo only                  |     0.9916 |  0.5939 | 0.1954 |     0.5157 |
| Serie A        | Logistic regression       |     0.9859 |  0.5902 | 0.1941 |     0.5277 |
| Serie A        | LightGBM                  |     0.9857 |  0.589  | 0.1937 |     0.5269 |
| Serie A        | Market average            |     0.9702 |  0.5788 | 0.1891 |     0.5284 |
| Serie A        | Pinnacle (opening)        |     0.9694 |  0.5783 | 0.189  |     0.5291 |
| Serie A        | Pinnacle (closing)        |     0.9675 |  0.5775 | 0.1886 |     0.5336 |
| La Liga        | Home-win prior (baseline) |     1.0662 |  0.6441 | 0.2264 |     0.4548 |
| La Liga        | Elo only                  |     0.9714 |  0.577  | 0.1944 |     0.5399 |
| La Liga        | Logistic regression       |     0.9724 |  0.5776 | 0.1949 |     0.5414 |
| La Liga        | LightGBM                  |     0.9678 |  0.575  | 0.1934 |     0.5444 |
| La Liga        | Market average            |     0.9585 |  0.5688 | 0.1907 |     0.5535 |
| La Liga        | Pinnacle (opening)        |     0.9581 |  0.5686 | 0.1906 |     0.552  |
| La Liga        | Pinnacle (closing)        |     0.9564 |  0.5677 | 0.19   |     0.552  |

## By season

|   season | model                     |   log_loss |   brier |    rps |   accuracy |
|---------:|:--------------------------|-----------:|--------:|-------:|-----------:|
|     2022 | Home-win prior (baseline) |     1.0629 |  0.6421 | 0.2291 |     0.4573 |
|     2022 | Elo only                  |     0.9896 |  0.589  | 0.2033 |     0.5361 |
|     2022 | Logistic regression       |     0.987  |  0.5877 | 0.2031 |     0.5329 |
|     2022 | LightGBM                  |     0.9857 |  0.5867 | 0.2021 |     0.5318 |
|     2022 | Market average            |     0.9779 |  0.5821 | 0.2    |     0.5411 |
|     2022 | Pinnacle (opening)        |     0.9775 |  0.5818 | 0.1999 |     0.5389 |
|     2022 | Pinnacle (closing)        |     0.9749 |  0.5802 | 0.1989 |     0.5345 |
|     2023 | Home-win prior (baseline) |     1.0777 |  0.6524 | 0.2289 |     0.4309 |
|     2023 | Elo only                  |     0.9753 |  0.5821 | 0.1945 |     0.5268 |
|     2023 | Logistic regression       |     0.9728 |  0.5797 | 0.1934 |     0.532  |
|     2023 | LightGBM                  |     0.974  |  0.5797 | 0.1935 |     0.5263 |
|     2023 | Market average            |     0.9556 |  0.5674 | 0.1879 |     0.5502 |
|     2023 | Pinnacle (opening)        |     0.955  |  0.5671 | 0.1877 |     0.5502 |
|     2023 | Pinnacle (closing)        |     0.9525 |  0.5657 | 0.1871 |     0.5588 |
|     2024 | Home-win prior (baseline) |     1.0802 |  0.6547 | 0.2337 |     0.4201 |
|     2024 | Elo only                  |     0.9854 |  0.5875 | 0.2008 |     0.5228 |
|     2024 | Logistic regression       |     0.9817 |  0.5855 | 0.1998 |     0.5291 |
|     2024 | LightGBM                  |     0.9821 |  0.5853 | 0.1999 |     0.5245 |
|     2024 | Market average            |     0.964  |  0.5731 | 0.1945 |     0.536  |
|     2024 | Pinnacle (opening)        |     0.9635 |  0.5728 | 0.1944 |     0.5371 |
|     2024 | Pinnacle (closing)        |     0.9606 |  0.5706 | 0.1935 |     0.5474 |
|     2025 | Home-win prior (baseline) |     1.0695 |  0.6466 | 0.2283 |     0.4465 |
|     2025 | Elo only                  |     0.9801 |  0.5843 | 0.198  |     0.5301 |
|     2025 | Logistic regression       |     0.9809 |  0.5843 | 0.1982 |     0.5334 |
|     2025 | LightGBM                  |     0.9796 |  0.5831 | 0.1974 |     0.539  |
|     2025 | Market average            |     0.9626 |  0.5724 | 0.1928 |     0.5412 |
|     2025 | Pinnacle (opening)        |     0.9628 |  0.5725 | 0.1928 |     0.5379 |
|     2025 | Pinnacle (closing)        |     0.9641 |  0.5736 | 0.1935 |     0.5379 |

## Paired bootstrap of log-loss differences (2,000 resamples)

- LightGBM - Pinnacle (closing): +0.0175 (95% CI +0.0136 to +0.0213)
- LightGBM - Pinnacle (opening): +0.0154 (95% CI +0.0117 to +0.0190)
- LightGBM - Market average: +0.0150 (95% CI +0.0113 to +0.0186)
- Logistic regression - Pinnacle (closing): +0.0176 (95% CI +0.0138 to +0.0217)
- Logistic regression - Pinnacle (opening): +0.0155 (95% CI +0.0119 to +0.0194)
- Logistic regression - Market average: +0.0151 (95% CI +0.0115 to +0.0190)
- Elo only - Pinnacle (closing): +0.0200 (95% CI +0.0157 to +0.0243)
- Elo only - Pinnacle (opening): +0.0179 (95% CI +0.0140 to +0.0219)
- Elo only - Market average: +0.0175 (95% CI +0.0137 to +0.0214)
- LightGBM - Logistic regression: -0.0001 (95% CI -0.0027 to +0.0025)
