import dash
from  dash import html
from dash import dcc

def render_dict(dictionary, indent_level=0):
    # Recursive function to render the dictionary
    def render_items(items, indent_level):
        rendered_items = []
        indent = "    " * indent_level  # Four spaces for each indentation level
        for key, value in items.items():
            if isinstance(value, dict):
                # If the value is a dictionary, recursively render it as a collapsible section
                rendered_value = render_items(value, indent_level + 1)
                rendered_items.append(html.Details([
                    html.Summary(f"{indent}{key}"),
                    html.Div(rendered_value)
                ]))
            else:
                # If the value is not a dictionary, render it as a simple key-value pair
                rendered_items.append(html.Div(f"{indent}{key}: {value}"))
        return rendered_items

    # Render the dictionary
    rendered_dict = render_items(dictionary, indent_level)

    # Create the Dash app layout
    app = dash.Dash(__name__)
    app.layout = html.Div(rendered_dict)

    # Run the app
    app.run_server(debug=True)

# Example dictionary
my_dict = {
    "name": "John Doe",
    "age": 30,
    "address": {
        "street": "123 Main Street",
        "city": "New York",
        "country": "USA"
    },
    "skills": {
        "programming": ["Python", "JavaScript"],
        "design": ["Photoshop", "Illustrator"]
    }
}

json_data = {
    "reactors": {
        "source": {
            "vessel": {
                "diameter": 1,
                "straight height length": 1,
                "bottom shape": "elliptical",
                "bottom depth": 0.25
            },
            "agitator": {
                "position": {
                    "x": 0,
                    "y": 0
                },
                "rotational direction": "clockwise",
                "impellers count": 1,
                "impellers": {
                    "impeller 1": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 2": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 3": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 4": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 5": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 6": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 7": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 8": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    }
                }
            },
            "diptubes": {
                "diptubes active": "false",
                "diptubes count": 1,
                "diptubes": {
                    "diptube 1": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 2": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 3": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 4": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 5": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 6": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 7": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 8": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    }
                }
            },
            "baffles": {
                "baffles active": "false",
                "baffles count": 4,
                "baffles": {
                    "baffle 1": {
                        "angle": 0,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 2": {
                        "angle": 90,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 3": {
                        "angle": 180,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 4": {
                        "angle": 270,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 5": {
                        "angle": 360,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 6": {
                        "angle": 450,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 7": {
                        "angle": 540,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 8": {
                        "angle": 630,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    }
                }
            },
            "operating conditions": {
                "operating conditions count": 2,
                "operating conditions": {
                    "operating condition 1": {
                        "volume": 333,
                        "stirring speed": 1
                    },
                    "operating condition 2": {
                        "volume": 566,
                        "stirring speed": 1
                    },
                    "operating condition 3": {
                        "volume": 800,
                        "stirring speed": 1
                    }
                }
            }
        },
        "target": {
            "vessel": {
                "diameter": 1,
                "straight height length": 1,
                "bottom shape": "elliptical",
                "bottom depth": 0.25
            },
            "agitator": {
                "position": {
                    "x": 0,
                    "y": 0
                },
                "rotational direction": "clockwise",
                "impellers count": 1,
                "impellers": {
                    "impeller 1": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 2": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 3": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 4": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 5": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 6": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 7": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    },
                    "impeller 8": {
                        "type": "centrifugal.stl",
                        "diameter": 0.25,
                        "axial position": 0
                    }
                }
            },
            "diptubes": {
                "diptubes active": "false",
                "diptubes count": 1,
                "diptubes": {
                    "diptube 1": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 2": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 3": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 4": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 5": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 6": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 7": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    },
                    "diptube 8": {
                        "diameter": 0.05,
                        "position": {
                            "x": 0.25,
                            "y": 0,
                            "z": 0
                        }
                    }
                }
            },
            "baffles": {
                "baffles active": "false",
                "baffles count": 4,
                "baffles": {
                    "baffle 1": {
                        "angle": 0,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 2": {
                        "angle": 90,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 3": {
                        "angle": 180,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 4": {
                        "angle": 270,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 5": {
                        "angle": 360,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 6": {
                        "angle": 450,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 7": {
                        "angle": 540,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    },
                    "baffle 8": {
                        "angle": 630,
                        "z": 0.1,
                        "width": 0.1,
                        "thickness": 0.01
                    }
                }
            },
            "operating conditions": {
                "operating conditions count": 2,
                "operating conditions": {
                    "operating condition 1": {
                        "volume": 333,
                        "stirring speed": 1.0
                    },
                    "operating condition 2": {
                        "volume": 566,
                        "stirring speed": 1.0
                    },
                    "operating condition 3": {
                        "volume": 800,
                        "stirring speed": 1.0
                    }
                }
            }
        }
    },
    "simulation settings": {
        "material": "user defined",
        "temperature": 200.0,
        "fluid properties": {
            "dynamic viscosity": 1,
            "density": 1
        },
        "mesh resolution": "medium"
    }
}


# Render the dictionary using Dash
render_dict(json_data)
