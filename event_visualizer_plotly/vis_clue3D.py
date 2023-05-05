import itertools
from functools import cached_property

import plotly.express as px
import plotly.graph_objects as go

#from hists.dataframe import *
import hists.parameters
from event_visualizer_plotly.utils import *

class LegendRanks:
    base = 1000

    tracksters = base + 100

    clus2D = tracksters + 100
    clus2D_discarded = clus2D + 10
    clus2D_chain = clus2D + 20

    rechits = clus2D + 100
    rechits_discarded = rechits - 10
    rechits_chain = rechits + 20


class Clue3DVisualization(BaseVisualization):
    def __init__(self, event:LoadedEvent) -> None:
        super().__init__(event)
        self.fig = create3DFigure(f"CLUE3D visualization - ntuple {event.record.ntupleNumber}, event {event.record.event} - e+ {event.record.beamEnergy} GeV")

        color_cycle = itertools.cycle(px.colors.qualitative.Plotly)
        self.mapClus3Did_color = NaNColorMap(
            {clus3D_id : next(color_cycle) for clus3D_id in self.event.clus3D_df.index.get_level_values("clus3D_id").drop_duplicates().to_list()},
            next(color_cycle))
        
        color_list = px.colors.qualitative.Dark24.copy()
        discarded_color = color_list.pop(5) # black
        color_cycle = itertools.cycle(color_list)
        self.mapClus2Did_color = NaNColorMap(
            {clus2D_id : next(color_cycle) for clus2D_id in self.event.clus2D_df.index.get_level_values("clus2D_id").drop_duplicates().to_list()},
            discarded_color)

        self.legendRanks = LegendRanks()

    def addSliders(self):
        updatemenu_kwargs = dict(xanchor="left", yanchor="top", y=1.3)
        xLeft=0.7
        self.fig.update_layout(updatemenus=[
            go.layout.Updatemenu(
                buttons=[
                    go.layout.updatemenu.Button(
                        label=f"Aspect ratio Z : {aspectRatioZ}",
                        method="relayout",
                        args=[{"scene.aspectratio.z" : aspectRatioZ} ]# the dot notation is needed
                    )
                for aspectRatioZ in [1., 2., 3., 4., 5., 7., 10.]],
                active=2, 
                x=xLeft,
                **updatemenu_kwargs
            ),
        ])
        return self

    def add3DClusters(self, groupAllTracksters=False):
        """ Add tracksters to plot
        Parameters : 
         - groupAllTracksters : if True, then a single legend entry will be shown for all tracksters. If False, each trackster has its individual legend entry
        """
        markerSizeScale = MarkerSizeLogScaler(self.event.clus3D_df.clus3D_energy, maxMarkerSize=60, minMarkerSize=30)
        def makeTrace(df:pd.DataFrame):
            if groupAllTracksters:
                legendArgs = dict(
                    name="Tracksters"
                )
            else:
                assert df.shape[0] == 1
                legendArgs = dict(
                    legendgroup="cluster3D",
                    legendgrouptitle_text="Tracksters",
                    name=f"Trackster nb {df.index[0]}") # Assume the input df has a single row

            self.fig.add_trace(go.Scatter3d(
                **legendArgs,
                legendrank=self.legendRanks.tracksters,
                x=df["clus3D_x"], y=df["clus3D_y"], z=df["clus3D_z"], 
                mode="markers",
                marker=dict(
                    symbol="cross",
                    color=df.index.to_series().map(self.mapClus3Did_color),
                    size=markerSizeScale.scale(df["clus3D_energy"]),
                ),
                customdata=np.stack((df.index.to_list(), df["clus3D_energy"], df["clus3D_size"]), axis=1),
                hovertemplate=
                "Trackster : %{customdata[0]}<br>"
                "Energy: %{customdata[1]:.3g} GeV<br>"
                "Size: %{customdata[2]}<br>"
                "x=%{x}, y=%{y}, z=%{z}<br>"
                "<extra></extra>",
                )
            )
        if groupAllTracksters:
            makeTrace(self.event.clus3D_df)
        else:
            for i in range(self.event.clus3D_df.shape[0]):
                # Make a Dataframe with a single row (as go.Scatter3D wants arrays as inputs, even when there is only a single scatter point)
                makeTrace(self.event.clus3D_df.iloc[i:i+1]) 
        return self

    def add2DClusters(self, chainAsIndividualGroup=False):
        """ Add 2D clusters and 2D clusters chain 
        Parameters :
         - chainAsIndividualGroup : if True, the chain of 2D clusters will be toggleable separately. If False, it is grouped with all 3D clusters
        """
        showLegend = True
        markerSizeScale = MarkerSizeLogScaler(self.event.clus2D_df.clus2D_energy, maxMarkerSize=14, minMarkerSize=3)
        for clus3D_id, grouped_df in self.event.clus2D_df.groupby("clus3D_id", dropna=False):
            if math.isnan(clus3D_id):
                trace_dict = dict(
                    marker_symbol=list(itertools.islice(self.clus3D_symbols_outlier_3Dview, grouped_df.shape[0])),
                    name="LC discarded in CLUE3D",
                    legendrank=self.legendRanks.clus2D_discarded,
                )
                
            else:
                trace_dict = dict(
                    marker_symbol=self.mapClus3Did_symbol_3Dview[clus3D_id],
                    name=f"LC in trackster nb {int(clus3D_id)}",
                    legendrank=self.legendRanks.clus2D,
                )
            
            self.fig.add_trace(go.Scatter3d(
                **trace_dict,
                mode="markers",
                legendgroup="cluster2D",
                legendgrouptitle_text="Layer clusters (LC)",
                x=grouped_df["clus2D_x"], y=grouped_df["clus2D_y"], z=grouped_df["clus2D_z"], 
                marker=dict(
                    color=self.event.clus2D_df.index.to_series().map(self.mapClus2Did_color),
                    line_color="black",
                    line_width=2, # Does not work on some graphics cards
                    size=markerSizeScale.scale(grouped_df["clus2D_energy"]),
                ),
                customdata=np.dstack((grouped_df.clus2D_energy, grouped_df.clus2D_rho, grouped_df.clus2D_delta,
                    grouped_df.clus2D_pointType.map({0:"Follower", 1:"Seed", 2:"Outlier"}),
                    grouped_df.clus2D_layer, grouped_df.clus2D_size))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "2D cluster : %{customdata[3]}<br>"
                    "Layer : %{customdata[4]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm<br>"
                    "Size: %{customdata[5]}"
                )
            ))

            if chainAsIndividualGroup:
                legend_kwargs = dict(
                    legendgroup="clus2D_chain"
                )
            else:
                legend_kwargs = dict(
                    legendgroup="cluster2D",
                )

            # dropna drops layer clusters which have nearestHigher = -1
            for row in grouped_df.dropna(subset="clus2D_x_ofNearestHigher").itertuples(index=False, name="Cluster2D"):
                if row.clus2D_pointType != 0:
                    # in CLUE3D layer clusters can have a nearestHigher but still be an outlier if distance to neareast higher is larger than outlierDeltaFactor * critical_transverse_distance
                    # Also a seed can have a nearestHigher set
                    # In these two cases do not draw the arrow
                    # Note that it is also possible for pointType == 1 (ie follower) but having no nearest higher
                    continue
                self.fig.add_traces(makeArrow3D(
                    row.clus2D_x, row.clus2D_x_ofNearestHigher, row.clus2D_y, row.clus2D_y_ofNearestHigher, row.clus2D_z, row.clus2D_z_ofNearestHigher,
                    dictLine=dict(
                        name="LC chain of nearest higher",
                        showlegend=showLegend,
                        legendrank=self.legendRanks.clus2D_chain, # Has to be above default (1000) so that it is shown after all LC
                        line_width=max(1, math.log(row.clus2D_cumulativeEnergy/0.1)), #line width in pixels
                    ), 
                    dictCone=dict(),
                    dictCombined=legend_kwargs,
                    color=self.mapClus3Did_color(clus3D_id),
                    )
                )

                showLegend = False
        return self
        
    def addRechits(self, hiddenByDefault=False, chainAsIndividualGroup=False):
        """ Add rechits and rechits chain 
        Parameters :
         - hiddenByDefault : should rechits (and chain) be hidden at startup (can still be enabled by clicking on legend)
         - chainAsIndividualGroup : if True, the chain of 2D clusters will be toggleable separately. If False, it is grouped with all 3D clusters
        """
        additional_trace_kwargs = dict()
        if hiddenByDefault:
            additional_trace_kwargs["visible"] = 'legendonly'
        showLegend = True
        markerSizeScale = MarkerSizeLogScaler(self.event.rechits_df.rechits_energy, maxMarkerSize=15, minMarkerSize=1)

        def mapClus2DidToProps(clus2D_id:int) -> dict:
            prop_dict = dict(
                marker_color=self.mapClus2Did_color(clus2D_id)
            )
            if math.isnan(clus2D_id):
                # can be 'circle', 'circle-open', 'cross', 'diamond', 'diamond-open', 'square', 'square-open', 'x'
                prop_dict["marker_symbol"] = "diamond"
                prop_dict["marker_opacity"] = 0.4
            else:
                prop_dict["marker_symbol"] = "circle"
                prop_dict["marker_opacity"] = 0.6
            return prop_dict

        for index, grouped_df in self.event.rechits_df.groupby(by=["clus3D_id", "clus2D_id"], dropna=False):
            if math.isnan(index[1]): # LC nb is NaN
                trace_dict = dict(
                    name="Discarded by CLUE",
                    legendrank=self.legendRanks.rechits_discarded, # Put at top of legend group
                )
            else:
                trace_dict = dict(
                    name=f"in LC cluster nb {index[1]}",
                    legendrank=self.legendRanks.rechits,
                )
            self.fig.add_trace(go.Scatter3d(
                mode="markers",
                legendgroup="rechits",
                legendgrouptitle_text="Rechits",
                **trace_dict,
                x=grouped_df["rechits_x"], y=grouped_df["rechits_y"], z=grouped_df["rechits_z"], 
                **mapClus2DidToProps(index[1]),
                marker=dict(
                    size=markerSizeScale.scale(grouped_df["rechits_energy"]),
                ),
                customdata=np.dstack((grouped_df.rechits_energy, grouped_df.rechits_rho, grouped_df.rechits_delta,
                    getPointTypeStringForRechits(clus2D_id=index[1], grouped_df=grouped_df), 
                    grouped_df.rechits_layer))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "Rechit : %{customdata[3]}<br>"
                    "Layer : %{customdata[4]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm"
                ),
                **additional_trace_kwargs
            ))

            if chainAsIndividualGroup:
                legend_kwargs = dict(
                    legendgroup="rechits_chain"
                )
            else:
                legend_kwargs = dict(
                    legendgroup="rechits",
                )
            
            for row in grouped_df.dropna(subset="rechits_x_ofNearestHigher").itertuples(index=False):
                if row.rechits_pointType != 0:
                    # Drop non-followers 
                    continue
                self.fig.add_traces(makeArrow3D(
                    row.rechits_x, row.rechits_x_ofNearestHigher, row.rechits_y, row.rechits_y_ofNearestHigher, row.rechits_z, row.rechits_z_ofNearestHigher,
                    dictLine=dict(
                        name="Rechits chain of nearest higher",
                        showlegend=showLegend,
                        legendrank=self.legendRanks.rechits_chain, # Has to be above default (1000) so that it is shown after all rechits
                        line_width=max(1, math.log(row.rechits_cumulativeEnergy/0.01)), #line width in pixels
                    ), 
                    dictCone=dict(),
                    dictCombined=additional_trace_kwargs | legend_kwargs | dict(opacity=0.5),
                    color=self.mapClus2Did_color(index[1]),
                    )
                )
                showLegend = False
        return self


    def addImpactTrajectory(self):
        impacts = self.event.impact_df
        self.fig.add_trace(go.Scatter3d(
            mode="lines",
            name="Impact from DWC",
            x=impacts.impactX, y=impacts.impactY, z=impacts.impactZ,
            line=dict(
                color="black",
                width=3,
            ),
            hoverinfo='skip',
        ))
        return self


    def addDetectorCylinder(self):
        """ Plot detector cylinder (very approximate detector area) """
        detExt = hists.parameters.DetectorExtentData # For data
        x, y, z = makeCylinderCoordinates(r=detExt.radius, h=detExt.depth, z0=detExt.firstLayerZ, axisX=detExt.centerX, axisY=detExt.centerY)
        self.fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale=[[0, 'blue'],[1, 'blue']], 
            opacity=0.5, hoverinfo="skip", showscale=False, showlegend=True, visible="legendonly",
            name="Approx detector size"))
        return self