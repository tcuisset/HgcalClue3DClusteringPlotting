import math

import awkward as ak
import pandas as pd
from dash import dash_table
from dash.dash_table.Format import Format, Scheme, Trim

from event_visualizer_plotly.utils import  LoadedEvent



def makeColumnsFromTuples(tuples:list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{}]

defaultIntegerFormat = dict(type="numeric")
defaultNbFormat = dict(type='numeric', format=Format(precision=3))

clus3DTable_columns = [
    dict(id="clus3D_id", name="Trackster ID", **defaultIntegerFormat),
    dict(id="clus3D_x", name="Trackster x (cm)", **defaultNbFormat),
    dict(id="clus3D_y", name="Trackster y (cm)", **defaultNbFormat),
    dict(id="clus3D_z", name="Trackster z (cm)", **defaultNbFormat),
    dict(id="clus3D_energy", name="Trackster energy (GeV)", **defaultNbFormat),
    dict(id="clus3D_size", name="Trackster size (ie nb of LC)", **defaultIntegerFormat),
]

datatables_settings = dict(filter_action='native', sort_action='native')

def makeClus3DTable() -> dash_table.DataTable:
    return dash_table.DataTable(id="clus3D_table", columns=clus3DTable_columns, **datatables_settings)
	
def updateClus3DTableData(event:LoadedEvent) -> tuple[dict, list[dict]]:
    """ Makes data, columns pair for Dash Datatable """
    return (event.clus3D_df
        .reset_index()
        .sort_values("clus3D_energy", ascending=False)
        [[col_dict["id"] for col_dict in clus3DTable_columns]]
        .to_dict('records')
    )

clus2DTable_columns = [
    dict(id="clus2D_id", name="LC ID", **defaultIntegerFormat),
    dict(id="clus2D_layer", name="LC layer", **defaultIntegerFormat),
    dict(id="clus3D_id_text", name="Trackster nb"),
    dict(id="clus2D_size", name="LC size (nb of rechits)", **defaultIntegerFormat),
    dict(id="clus2D_rho", name="LC rho (local energy density for CLUE3D, GeV)", **defaultNbFormat),
    dict(id="clus2D_delta", name="LC delta (distance to nearest higher for CLUE3D, cm)", **defaultNbFormat),
    dict(id="clus2D_pointType_text", name="LC point type for CLUE3D"),
    dict(id="clus2D_nearestHigher", name="LC nearest higher ID (CLUE3D)", **defaultIntegerFormat),
]

def makeClus2DTable() -> dash_table.DataTable:
    return dash_table.DataTable(id="clus2D_table", columns=clus2DTable_columns, **datatables_settings)
	

def updateClus2DTableData(event:LoadedEvent) -> tuple[dict, list[dict]]:
    """ Makes data, columns pair for Dash Datatable """
    df = (event.clus2D_df
        [["clus2D_layer", "clus2D_energy", "clus3D_id", "clus2D_size", "clus2D_rho", "clus2D_delta", "clus2D_pointType", "clus2D_nearestHigher"]]
        .reset_index()
        .sort_values(["clus2D_layer", "clus2D_energy"], ascending=[True, False])
    )
    return (df
        .assign(
            clus2D_pointType_text=df.clus2D_pointType.map({0:"Follower", 1:"Seed", 2:"Outlier"}),
            clus3D_id_text=df.clus3D_id.replace({math.nan:"Not in a trackster"})
        )
        .drop(["clus2D_pointType", "clus3D_id"], axis="columns")
        .to_dict('records')
    )