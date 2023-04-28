import itertools
from functools import cached_property

import plotly.express as px
import plotly.graph_objects as go

#from hists.dataframe import *
import hists.parameters
from event_visualizer_plotly.utils import *


class Clue3DVisualization(BaseVisualization):
    def __init__(self, event:LoadedEvent) -> None:
        super().__init__(event)
        self.fig = create3DFigure(f"CLUE3D visualization - ntuple {event.record.ntupleNumber}, event {event.record.event} - e+ {event.record.beamEnergy} GeV")

        color_cycle = itertools.cycle(px.colors.qualitative.Plotly)
        self.mapClus3Did_color = NaNColorMap(
            {clus3D_id : next(color_cycle) for clus3D_id in self.clus3D_df.index.get_level_values("clus3D_id").drop_duplicates().to_list()},
            next(color_cycle))
        self.mapClus2Did_color = NaNColorMap(
            {clus2D_id : next(color_cycle) for clus2D_id in self.clus2D_df.index.get_level_values("clus2D_id").drop_duplicates().to_list()},
            next(color_cycle))

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
            go.layout.Updatemenu(
                buttons=[
                    go.layout.updatemenu.Button(
                        label=f"Outer rim width: {outerRimWidth}",
                        method="restyle", # We should use the traceIndices parameter as well for performance : https://plotly.com/javascript/plotlyjs-function-reference/#plotlyrestyle
                        args=[{"marker.line.width" : outerRimWidth}], # the dot notation is needed
                    )
                for outerRimWidth in [1, 3, 5, 7, 10, 15, 20, 40, 100, 300]],
                active=1,
                x=xLeft+0.15,
                **updatemenu_kwargs
            ),
        ])
        return self

    def add3DClusters(self):
        markerSizeScale = MarkerSizeLogScaler(self.clus3D_df.clus3D_energy, maxMarkerSize=60, minMarkerSize=30)
        self.fig.add_trace(go.Scatter3d(
            name="3D tracksters",
            x=self.clus3D_df["clus3D_x"], y=self.clus3D_df["clus3D_y"], z=self.clus3D_df["clus3D_z"], 
            mode="markers",
            marker=dict(
                symbol="cross",
                color=self.clus3D_df.index.to_series().map(self.mapClus3Did_color),
                size=markerSizeScale.scale(self.clus3D_df["clus3D_energy"]),
            ),
            customdata=np.dstack((self.clus3D_df["clus3D_energy"], self.clus3D_df["clus3D_size"]))[0],
            hovertemplate="clus3D_energy=%{customdata[0]:.3g} GeV<br>clus3D_x=%{x}<br>clus3D_y=%{y}<br>clus3D_z=%{z}<br>clus3D_size=%{customdata[1]}<extra></extra>",
            )
        )
        return self

    def add2DClusters(self):
        showLegend = True
        markerSizeScale = MarkerSizeLogScaler(self.clus2D_df.clus2D_energy, maxMarkerSize=30)
        for clus3D_id, grouped_df in self.clus2D_df.groupby("clus3D_id", dropna=False):
            self.fig.add_trace(go.Scatter3d(
                mode="markers",
                legendgroup="cluster2D",
                legendgrouptitle_text="2D clusters",
                name=f"Trackster nb {clus3D_id}",
                x=grouped_df["clus2D_x"], y=grouped_df["clus2D_y"], z=grouped_df["clus2D_z"], 
                marker=dict(
                    symbol="circle",
                    color=self.clus2D_df.index.to_series().map(self.mapClus2Did_color),
                    #size=grouped_df["clus2D_size"],
                    size=markerSizeScale.scale(grouped_df["clus2D_energy"]),
                    line=dict(
                        color=self.mapClus3Did_color(clus3D_id),
                        width=3
                    ),
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
                        name="Cluster 2D chain",
                        legendgroup="clus2D_chain",
                        showlegend=showLegend,
                        line_width=max(1, math.log(row.clus2D_cumulativeEnergy/0.1)), #line width in pixels
                    ), dictCone=dict(
                        legendgroup="clus2D_chain"
                    ),
                    color=self.mapClus3Did_color(clus3D_id),
                    )
                )

                showLegend = False
        return self
        
    def addRechits(self):
        showLegend = True
        markerSizeScale = MarkerSizeLogScaler(self.rechits_df.rechits_energy, maxMarkerSize=25, minMarkerSize=1)
        for index, grouped_df in self.rechits_df.groupby(by=["clus3D_id", "clus2D_id"], dropna=False):
            self.fig.add_trace(go.Scatter3d(
                mode="markers",
                legendgroup="rechits",
                legendgrouptitle_text="Rechits",
                name=f"2D cluster nb {index[1]}",
                x=grouped_df["rechits_x"], y=grouped_df["rechits_y"], z=grouped_df["rechits_z"], 
                marker=dict(
                    symbol="circle",
                    color=self.mapClus2Did_color(index[1]),
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
                )
            ))
            for row in grouped_df.dropna(subset="rechits_x_ofNearestHigher").itertuples(index=False):
                if row.rechits_pointType != 0:
                    # Drop non-followers 
                    continue
                self.fig.add_traces(makeArrow3D(
                    row.rechits_x, row.rechits_x_ofNearestHigher, row.rechits_y, row.rechits_y_ofNearestHigher, row.rechits_z, row.rechits_z_ofNearestHigher,
                    dictLine=dict(
                        name="Rechits chain",
                        legendgroup="rechits_chain",
                        showlegend=showLegend,
                        line_width=max(1, math.log(row.rechits_cumulativeEnergy/0.01)), #line width in pixels
                    ), dictCone=dict(
                        legendgroup="rechits_chain"
                    ),
                    color=self.mapClus2Did_color(index[1]),
                    )
                )
                showLegend = False
        return self


    def addImpactTrajectory(self):
        impacts = self.impact_df
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