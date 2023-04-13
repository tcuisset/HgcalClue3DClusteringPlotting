import itertools
from functools import cached_property

import plotly.express as px
import plotly.graph_objects as go

from hists.dataframe import *
from utils import *




class LayerVisualization(BaseVisualization):
    def __init__(self, comp:DataframeComputations, eventNb, layerNb) -> None:
        super().__init__(comp, eventNb)
        self.layerNb = layerNb
        self.fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"Layer {layerNb} view"),
                #width=1200,
                #height=600,
                autosize=True
            )
        )

        color_cycle = itertools.cycle(px.colors.qualitative.Plotly)
        self.mapClus3Did_color = {clus3D_id : next(color_cycle) for clus3D_id in self.clus3D_df.index.get_level_values(0).drop_duplicates().to_list()}
        self.mapClus2Did_color = {clus2D_id : next(color_cycle) for clus2D_id in self.clus2D_df.index.get_level_values("clus2D_id").drop_duplicates().to_list()}

    @property
    def rechits_df_onLayer(self):
        return self.rechits_df[self.rechits_df.rechits_layer == self.layerNb]
    
    @property
    def clus2D_df_onLayer(self):
        return self.clus2D_df[self.clus2D_df.clus2D_layer == self.layerNb]
    
    def add2DClusters(self):
        showLegend = True
        for clus3D_id, grouped_df in self.clus2D_df_onLayer.groupby("clus3D_id"):
            self.fig.add_trace(go.Scatter(
                mode="markers",
                legendgroup="cluster2D",
                legendgrouptitle_text="2D clusters",
                name=f"Trackster nb {clus3D_id}",
                x=grouped_df["clus2D_x"], y=grouped_df["clus2D_y"],
                marker=dict(
                    symbol="circle",
                    color=self.clus2D_df_onLayer.index.get_level_values(level="clus2D_id").map(self.mapClus2Did_color),
                    size=grouped_df["clus2D_size"],
                    line=dict(
                        color=self.mapClus3Did_color[clus3D_id],
                        width=3.
                    )
                ),
                customdata=np.dstack((grouped_df.clus2D_energy, grouped_df.clus2D_rho, grouped_df.clus2D_delta,
                    grouped_df.clus2D_pointType.map({0:"Follower", 1:"Seed", 2:"Outlier"})))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "2D cluster : %{customdata[3]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm<br>"
                    "Size: %{marker.size}"
                )
            ))

            # for row in grouped_df.dropna().itertuples(index=False, name="Cluster2D"):
            #     x1, x2, y1, y2, z1, z2 = row.clus2D_x, row.clus2D_x_ofNearestHigher, row.clus2D_y, row.clus2D_y_ofNearestHigher, row.clus2D_z, row.clus2D_z_ofNearestHigher
            #     self.fig.add_traces(go.Scatter(
            #         mode="lines+markers",
            #         name="Cluster 2D chain",
            #         showlegend=showLegend,
            #         hoverinfo='skip',
            #         x=[x1, x2],
            #         y=[y1, y2],
            #         marker=dict(
            #             symbol="arrow",
            #             color=self.mapClus3Did_color[clus3D_id],
            #         ),
            #         line_width=max(1, math.log(row.clus2D_cumulativeEnergy/0.1)), #line width in pixels
            #     ))

            #     showLegend = False
        return self
        
    def addRechits(self):
        showLegend = True
        for index, grouped_df in self.rechits_df_onLayer.groupby(level=["clus3D_id", "clus2D_id"]):
            self.fig.add_trace(go.Scatter(
                mode="markers",
                legendgroup="rechits",
                legendgrouptitle_text="Rechits",
                name=f"2D cluster nb {index[1]}",
                x=grouped_df["rechits_x"], y=grouped_df["rechits_y"], 
                marker=dict(
                    symbol="circle",
                    color=self.mapClus2Did_color[index[1]],
                    size=np.log(grouped_df["rechits_energy"]/0.00001).clip(lower=1),
                ),
                customdata=np.dstack((grouped_df.rechits_energy, grouped_df.rechits_rho, grouped_df.rechits_delta,
                    grouped_df.rechits_pointType.map({0:"Follower", 1:"Seed", 2:"Outlier"})))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "Rechit : %{customdata[3]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm"
                )
            ))
            for row in grouped_df.dropna().itertuples(index=False):
                x1, x2, y1, y2 = row.rechits_x, row.rechits_x_ofNearestHigher, row.rechits_y, row.rechits_y_ofNearestHigher,
                self.fig.add_traces(go.Scatter(
                    mode="markers+lines",
                    name="Rechits chain",
                    legendgroup="rechits_chain",
                    showlegend=showLegend,
                    hoverinfo='skip',
                    x=[x1, x2],
                    y=[y1, y2],
                    marker=dict(
                        symbol="arrow",
                        color=self.mapClus2Did_color[index[1]],
                        size=10,
                        angleref="previous",
                        #standoff=8,
                    ),
                    line_width=max(1, math.log(row.rechits_cumulativeEnergy/0.01)), #line width in pixels
                ))
                showLegend = False

        return self

    # def addCircleSearchForComputingClusterPosition(self, thresholdW0):
    #     """ Draw a circle around each 2D cluster representing the search area for rechits to compute the 2D cluster position.
    #     ie a circle centered on the maximum energy cell of the cluster and of radius thresholdW0 """
