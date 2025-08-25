import json
import asyncio
from typing import AsyncGenerator

async def mock_suggestion() -> AsyncGenerator[str, None]:

  """[테스트용] 최종 확인에 대한 mock 응답"""
  message = '''
이것은 테스트용 csv url 입니다. [csv 다운로드](https://rgmarketaiagentb767.blob.core.windows.net/minti-images/ad_daily.csv?sp=r&st=2025-08-25T06:38:34Z&se=2025-08-31T14:53:34Z&skoid=03d5b0e0-130c-4804-8b73-56ec3a3ff135&sktid=736d39e1-5c76-403f-9148-7432afb3f83b&skt=2025-08-25T06:38:34Z&ske=2025-08-31T14:53:34Z&sks=b&skv=2024-11-04&sv=2024-11-04&sr=b&sig=XyoY6G8HT1BErh1kA2Ya21nWUjqC4UHnJuEKpPirckM%3D)
최종 확인 테스트입니다. 프로모션 계획이 완성되었습니다. 다음 단계로 진행하시겠습니까?
'''
  payload = {"type": "start"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
  
  # 스트리밍 형태로 응답
  for char in message:
      payload = {
          "type": "chunk",
          "content": char
      }
      yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
      await asyncio.sleep(0.02)  # 타이핑 효과

  payload = {"type": "table_start"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

  Table = '''
| product_id | product_name | brand | category_l1 | category_l2 | total_purchase_count |
|---|---|---|---|---|---|
| 26193 | 트리거포인트 MBX 마사지볼 | 트리거포인트 | 헬스,건강용품 | 마사지,보호대 | 15.0 |
| 5234 | 조르단 어린이 버디1 1P (색상랜덤) | 조르단 | 구강용품 | 칫솔 | 14.0 |
| 8052 | 프레쉬라이트 폼 쿨블루 (염색) | 프레쉬라이트 | 헤어케어 | 염색약,펌 | 14.0 |
| 11952 | 디어러스 쉬어 네일 10종 | 디어러스 | 네일 | 일반네일 | 14.0 |
| 26680 | 아베다 스칼프솔루션 리프레싱 프로텍티브 미스트 100ml | 아베다 | 헤어케어 | 트리트먼트,팩 | 14.0 |
'''

  for char in Table:
      payload = {
          "type": "chunk",
          "content": char
      }
      yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
      await asyncio.sleep(0.02)  # 타이핑 효과

  payload = {"type": "table_end"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

  example_graph = {
    "data": [
        {
          "alignmentgroup": "True",
          "hovertemplate": "상품명=%{x}<br>구매 횟수=%{marker.color}<extra></extra>",
          "legendgroup": "",
          "marker": {
            "color": [
              21.0,
              18.0,
              18.0,
              17.0,
              17.0
            ],
            "coloraxis": "coloraxis",
            "pattern": {
              "shape": ""
            }
          },
          "name": "",
          "offsetgroup": "",
          "orientation": "v",
          "showlegend": false,
          "textposition": "auto",
          "x": [
            "삼대오백 매시브 오리지널 그립 3종 택1 (S/M/L)",
            "하빈져 파워 글러브 헬스 장갑 여성용 (멜롯)",
            "[리뉴얼] 스킨푸드 블랙슈가 퍼펙트 토너 2X 포 맨 180ml",
            "[가정상비 필수오일] 프라나롬 라빈트사라 오일 10ml",
            "아이캔디 레인보우 볼륨 S 브러쉬 미니 블랙"
          ],
          "xaxis": "x",
          "y": [
            21.0,
            18.0,
            18.0,
            17.0,
            17.0
          ],
          "yaxis": "y",
          "type": "bar"
        }
      ],
      "layout": {
        "template": {
          "data": {
            "histogram2dcontour": [
              {
                "type": "histogram2dcontour",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                },
                "colorscale": [
                  [
                    0.0,
                    "#0d0887"
                  ],
                  [
                    0.1111111111111111,
                    "#46039f"
                  ],
                  [
                    0.2222222222222222,
                    "#7201a8"
                  ],
                  [
                    0.3333333333333333,
                    "#9c179e"
                  ],
                  [
                    0.4444444444444444,
                    "#bd3786"
                  ],
                  [
                    0.5555555555555556,
                    "#d8576b"
                  ],
                  [
                    0.6666666666666666,
                    "#ed7953"
                  ],
                  [
                    0.7777777777777778,
                    "#fb9f3a"
                  ],
                  [
                    0.8888888888888888,
                    "#fdca26"
                  ],
                  [
                    1.0,
                    "#f0f921"
                  ]
                ]
              }
            ],
            "choropleth": [
              {
                "type": "choropleth",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                }
              }
            ],
            "histogram2d": [
              {
                "type": "histogram2d",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                },
                "colorscale": [
                  [
                    0.0,
                    "#0d0887"
                  ],
                  [
                    0.1111111111111111,
                    "#46039f"
                  ],
                  [
                    0.2222222222222222,
                    "#7201a8"
                  ],
                  [
                    0.3333333333333333,
                    "#9c179e"
                  ],
                  [
                    0.4444444444444444,
                    "#bd3786"
                  ],
                  [
                    0.5555555555555556,
                    "#d8576b"
                  ],
                  [
                    0.6666666666666666,
                    "#ed7953"
                  ],
                  [
                    0.7777777777777778,
                    "#fb9f3a"
                  ],
                  [
                    0.8888888888888888,
                    "#fdca26"
                  ],
                  [
                    1.0,
                    "#f0f921"
                  ]
                ]
              }
            ],
            "heatmap": [
              {
                "type": "heatmap",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                },
                "colorscale": [
                  [
                    0.0,
                    "#0d0887"
                  ],
                  [
                    0.1111111111111111,
                    "#46039f"
                  ],
                  [
                    0.2222222222222222,
                    "#7201a8"
                  ],
                  [
                    0.3333333333333333,
                    "#9c179e"
                  ],
                  [
                    0.4444444444444444,
                    "#bd3786"
                  ],
                  [
                    0.5555555555555556,
                    "#d8576b"
                  ],
                  [
                    0.6666666666666666,
                    "#ed7953"
                  ],
                  [
                    0.7777777777777778,
                    "#fb9f3a"
                  ],
                  [
                    0.8888888888888888,
                    "#fdca26"
                  ],
                  [
                    1.0,
                    "#f0f921"
                  ]
                ]
              }
            ],
            "heatmapgl": [
              {
                "type": "heatmapgl",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                },
                "colorscale": [
                  [
                    0.0,
                    "#0d0887"
                  ],
                  [
                    0.1111111111111111,
                    "#46039f"
                  ],
                  [
                    0.2222222222222222,
                    "#7201a8"
                  ],
                  [
                    0.3333333333333333,
                    "#9c179e"
                  ],
                  [
                    0.4444444444444444,
                    "#bd3786"
                  ],
                  [
                    0.5555555555555556,
                    "#d8576b"
                  ],
                  [
                    0.6666666666666666,
                    "#ed7953"
                  ],
                  [
                    0.7777777777777778,
                    "#fb9f3a"
                  ],
                  [
                    0.8888888888888888,
                    "#fdca26"
                  ],
                  [
                    1.0,
                    "#f0f921"
                  ]
                ]
              }
            ],
            "contourcarpet": [
              {
                "type": "contourcarpet",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                }
              }
            ],
            "contour": [
              {
                "type": "contour",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                },
                "colorscale": [
                  [
                    0.0,
                    "#0d0887"
                  ],
                  [
                    0.1111111111111111,
                    "#46039f"
                  ],
                  [
                    0.2222222222222222,
                    "#7201a8"
                  ],
                  [
                    0.3333333333333333,
                    "#9c179e"
                  ],
                  [
                    0.4444444444444444,
                    "#bd3786"
                  ],
                  [
                    0.5555555555555556,
                    "#d8576b"
                  ],
                  [
                    0.6666666666666666,
                    "#ed7953"
                  ],
                  [
                    0.7777777777777778,
                    "#fb9f3a"
                  ],
                  [
                    0.8888888888888888,
                    "#fdca26"
                  ],
                  [
                    1.0,
                    "#f0f921"
                  ]
                ]
              }
            ],
            "surface": [
              {
                "type": "surface",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                },
                "colorscale": [
                  [
                    0.0,
                    "#0d0887"
                  ],
                  [
                    0.1111111111111111,
                    "#46039f"
                  ],
                  [
                    0.2222222222222222,
                    "#7201a8"
                  ],
                  [
                    0.3333333333333333,
                    "#9c179e"
                  ],
                  [
                    0.4444444444444444,
                    "#bd3786"
                  ],
                  [
                    0.5555555555555556,
                    "#d8576b"
                  ],
                  [
                    0.6666666666666666,
                    "#ed7953"
                  ],
                  [
                    0.7777777777777778,
                    "#fb9f3a"
                  ],
                  [
                    0.8888888888888888,
                    "#fdca26"
                  ],
                  [
                    1.0,
                    "#f0f921"
                  ]
                ]
              }
            ],
            "mesh3d": [
              {
                "type": "mesh3d",
                "colorbar": {
                  "outlinewidth": 0,
                  "ticks": ""
                }
              }
            ],
            "scatter": [
              {
                "fillpattern": {
                  "fillmode": "overlay",
                  "size": 10,
                  "solidity": 0.2
                },
                "type": "scatter"
              }
            ],
            "parcoords": [
              {
                "type": "parcoords",
                "line": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "scatterpolargl": [
              {
                "type": "scatterpolargl",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "bar": [
              {
                "error_x": {
                  "color": "#2a3f5f"
                },
                "error_y": {
                  "color": "#2a3f5f"
                },
                "marker": {
                  "line": {
                    "color": "#E5ECF6",
                    "width": 0.5
                  },
                  "pattern": {
                    "fillmode": "overlay",
                    "size": 10,
                    "solidity": 0.2
                  }
                },
                "type": "bar"
              }
            ],
            "scattergeo": [
              {
                "type": "scattergeo",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "scatterpolar": [
              {
                "type": "scatterpolar",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "histogram": [
              {
                "marker": {
                  "pattern": {
                    "fillmode": "overlay",
                    "size": 10,
                    "solidity": 0.2
                  }
                },
                "type": "histogram"
              }
            ],
            "scattergl": [
              {
                "type": "scattergl",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "scatter3d": [
              {
                "type": "scatter3d",
                "line": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                },
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "scattermapbox": [
              {
                "type": "scattermapbox",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "scatterternary": [
              {
                "type": "scatterternary",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "scattercarpet": [
              {
                "type": "scattercarpet",
                "marker": {
                  "colorbar": {
                    "outlinewidth": 0,
                    "ticks": ""
                  }
                }
              }
            ],
            "carpet": [
              {
                "aaxis": {
                  "endlinecolor": "#2a3f5f",
                  "gridcolor": "white",
                  "linecolor": "white",
                  "minorgridcolor": "white",
                  "startlinecolor": "#2a3f5f"
                },
                "baxis": {
                  "endlinecolor": "#2a3f5f",
                  "gridcolor": "white",
                  "linecolor": "white",
                  "minorgridcolor": "white",
                  "startlinecolor": "#2a3f5f"
                },
                "type": "carpet"
              }
            ],
            "table": [
              {
                "cells": {
                  "fill": {
                    "color": "#EBF0F8"
                  },
                  "line": {
                    "color": "white"
                  }
                },
                "header": {
                  "fill": {
                    "color": "#C8D4E3"
                  },
                  "line": {
                    "color": "white"
                  }
                },
                "type": "table"
              }
            ],
            "barpolar": [
              {
                "marker": {
                  "line": {
                    "color": "#E5ECF6",
                    "width": 0.5
                  },
                  "pattern": {
                    "fillmode": "overlay",
                    "size": 10,
                    "solidity": 0.2
                  }
                },
                "type": "barpolar"
              }
            ],
            "pie": [
              {
                "automargin": true,
                "type": "pie"
              }
            ]
          },
          "layout": {
            "autotypenumbers": "strict",
            "colorway": [
              "#636efa",
              "#EF553B",
              "#00cc96",
              "#ab63fa",
              "#FFA15A",
              "#19d3f3",
              "#FF6692",
              "#B6E880",
              "#FF97FF",
              "#FECB52"
            ],
            "font": {
              "color": "#2a3f5f"
            },
            "hovermode": "closest",
            "hoverlabel": {
              "align": "left"
            },
            "paper_bgcolor": "white",
            "plot_bgcolor": "#E5ECF6",
            "polar": {
              "bgcolor": "#E5ECF6",
              "angularaxis": {
                "gridcolor": "white",
                "linecolor": "white",
                "ticks": ""
              },
              "radialaxis": {
                "gridcolor": "white",
                "linecolor": "white",
                "ticks": ""
              }
            },
            "ternary": {
              "bgcolor": "#E5ECF6",
              "aaxis": {
                "gridcolor": "white",
                "linecolor": "white",
                "ticks": ""
              },
              "baxis": {
                "gridcolor": "white",
                "linecolor": "white",
                "ticks": ""
              },
              "caxis": {
                "gridcolor": "white",
                "linecolor": "white",
                "ticks": ""
              }
            },
            "coloraxis": {
              "colorbar": {
                "outlinewidth": 0,
                "ticks": ""
              }
            },
            "colorscale": {
              "sequential": [
                [
                  0.0,
                  "#0d0887"
                ],
                [
                  0.1111111111111111,
                  "#46039f"
                ],
                [
                  0.2222222222222222,
                  "#7201a8"
                ],
                [
                  0.3333333333333333,
                  "#9c179e"
                ],
                [
                  0.4444444444444444,
                  "#bd3786"
                ],
                [
                  0.5555555555555556,
                  "#d8576b"
                ],
                [
                  0.6666666666666666,
                  "#ed7953"
                ],
                [
                  0.7777777777777778,
                  "#fb9f3a"
                ],
                [
                  0.8888888888888888,
                  "#fdca26"
                ],
                [
                  1.0,
                  "#f0f921"
                ]
              ],
              "sequentialminus": [
                [
                  0.0,
                  "#0d0887"
                ],
                [
                  0.1111111111111111,
                  "#46039f"
                ],
                [
                  0.2222222222222222,
                  "#7201a8"
                ],
                [
                  0.3333333333333333,
                  "#9c179e"
                ],
                [
                  0.4444444444444444,
                  "#bd3786"
                ],
                [
                  0.5555555555555556,
                  "#d8576b"
                ],
                [
                  0.6666666666666666,
                  "#ed7953"
                ],
                [
                  0.7777777777777778,
                  "#fb9f3a"
                ],
                [
                  0.8888888888888888,
                  "#fdca26"
                ],
                [
                  1.0,
                  "#f0f921"
                ]
              ],
              "diverging": [
                [
                  0,
                  "#8e0152"
                ],
                [
                  0.1,
                  "#c51b7d"
                ],
                [
                  0.2,
                  "#de77ae"
                ],
                [
                  0.3,
                  "#f1b6da"
                ],
                [
                  0.4,
                  "#fde0ef"
                ],
                [
                  0.5,
                  "#f7f7f7"
                ],
                [
                  0.6,
                  "#e6f5d0"
                ],
                [
                  0.7,
                  "#b8e186"
                ],
                [
                  0.8,
                  "#7fbc41"
                ],
                [
                  0.9,
                  "#4d9221"
                ],
                [
                  1,
                  "#276419"
                ]
              ]
            },
            "xaxis": {
              "gridcolor": "white",
              "linecolor": "white",
              "ticks": "",
              "title": {
                "standoff": 15
              },
              "zerolinecolor": "white",
              "automargin": true,
              "zerolinewidth": 2
            },
            "yaxis": {
              "gridcolor": "white",
              "linecolor": "white",
              "ticks": "",
              "title": {
                "standoff": 15
              },
              "zerolinecolor": "white",
              "automargin": true,
              "zerolinewidth": 2
            },
            "scene": {
              "xaxis": {
                "backgroundcolor": "#E5ECF6",
                "gridcolor": "white",
                "linecolor": "white",
                "showbackground": true,
                "ticks": "",
                "zerolinecolor": "white",
                "gridwidth": 2
              },
              "yaxis": {
                "backgroundcolor": "#E5ECF6",
                "gridcolor": "white",
                "linecolor": "white",
                "showbackground": true,
                "ticks": "",
                "zerolinecolor": "white",
                "gridwidth": 2
              },
              "zaxis": {
                "backgroundcolor": "#E5ECF6",
                "gridcolor": "white",
                "linecolor": "white",
                "showbackground": true,
                "ticks": "",
                "zerolinecolor": "white",
                "gridwidth": 2
              }
            },
            "shapedefaults": {
              "line": {
                "color": "#2a3f5f"
              }
            },
            "annotationdefaults": {
              "arrowcolor": "#2a3f5f",
              "arrowhead": 0,
              "arrowwidth": 1
            },
            "geo": {
              "bgcolor": "white",
              "landcolor": "#E5ECF6",
              "subunitcolor": "white",
              "showland": true,
              "showlakes": true,
              "lakecolor": "white"
            },
            "title": {
              "x": 0.05
            },
            "mapbox": {
              "style": "light"
            }
          }
        },
        "xaxis": {
          "anchor": "y",
          "domain": [
            0.0,
            1.0
          ],
          "title": {
            "text": "상품명"
          },
          "type": "category",
          "tickangle": -45
        },
        "yaxis": {
          "anchor": "x",
          "domain": [
            0.0,
            1.0
          ],
          "title": {
            "text": "구매 횟수"
          }
        },
        "coloraxis": {
          "colorbar": {
            "title": {
              "text": "구매 횟수"
            }
          },
          "colorscale": [
            [
              0.0,
              "#0d0887"
            ],
            [
              0.1111111111111111,
              "#46039f"
            ],
            [
              0.2222222222222222,
              "#7201a8"
            ],
            [
              0.3333333333333333,
              "#9c179e"
            ],
            [
              0.4444444444444444,
              "#bd3786"
            ],
            [
              0.5555555555555556,
              "#d8576b"
            ],
            [
              0.6666666666666666,
              "#ed7953"
            ],
            [
              0.7777777777777778,
              "#fb9f3a"
            ],
            [
              0.8888888888888888,
              "#fdca26"
            ],
            [
              1.0,
              "#f0f921"
            ]
          ]
        },
        "legend": {
          "tracegroupgap": 0
        },
        "title": {
          "text": "30대 여성이 많이 구매한 상품 Top 5"
        },
        "barmode": "relative",
        "margin": {
          "l": 50,
          "r": 50,
          "t": 80,
          "b": 150
        }
      }
    }
  payload = {"type": "plan", "content": "brand"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

  payload = {'type': "graph", "content": example_graph}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    

    # 완료 신호
  payload = {"type": "done"}
  yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"