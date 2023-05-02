import itertools
from functools import cached_property

import plotly.express as px
import plotly.graph_objects as go

import hists.parameters
from event_visualizer_plotly.utils import *


class LayerVisualization(BaseVisualization):
    def __init__(self, event:LoadedEvent, layerNb:int) -> None:
        super().__init__(event)
        self.layerNb = layerNb
        self.fig = go.Figure(
            layout=go.Layout(
                title=go.layout.Title(text=f"Layer visualization - ntuple {event.record.ntupleNumber}, event {event.record.event} - e+ {event.record.beamEnergy} GeV - Layer {layerNb}"),
                #width=1200,
                #height=600,
                autosize=True
            )
        )
        self.fig.update_yaxes( # Same scale for x and y
            scaleanchor="x",
            scaleratio=1,
        )

        color_cycle = itertools.cycle(px.colors.qualitative.Plotly)
        self.mapClus2Did_color = NaNColorMap(
            {clus2D_id : next(color_cycle) for clus2D_id in self.clus2D_ids},
            next(color_cycle)
        )

    @property
    def rechits_df_onLayer(self) -> pd.DataFrame:
        return self.rechits_df[self.rechits_df.rechits_layer == self.layerNb]
    
    @property
    def clus2D_df_onLayer(self) -> pd.DataFrame:
        return self.clus2D_df[self.clus2D_df.clus2D_layer == self.layerNb]
    
    @cached_property
    def totalEnergyOnLayer(self) -> float:
        return self.rechits_df_onLayer.rechits_energy.sum()
    @cached_property
    def maxRechitEnergyOnLayer(self) -> float:
        return self.rechits_df_onLayer.rechits_energy.max()
    
    def add2DClusters(self):
        def scaleClus2DMarkerSize(size):
            return np.clip(size*3, 8, 70, )
        for clus3D_id, grouped_df in self.clus2D_df_onLayer.groupby("clus3D_id", dropna=False):
            if math.isnan(clus3D_id):
                clus3D_id_symbol = next(self.clus3D_symbols_outlier_2Dview)
            else:
                clus3D_id_symbol = self.mapClus3Did_symbol_2Dview[clus3D_id]
            
            self.fig.add_trace(go.Scatter(
                mode="markers",
                legendgroup="cluster2D",
                legendgrouptitle_text="2D clusters",
                name=f"Trackster nb {clus3D_id}",
                x=grouped_df["clus2D_x"], y=grouped_df["clus2D_y"],
                marker=dict(
                    symbol=clus3D_id_symbol,
                    color=grouped_df.index.map(self.mapClus2Did_color),
                    line_color="black",
                    line_width=2, # Does not work on some graphics cards
                    size=scaleClus2DMarkerSize(grouped_df["clus2D_size"]),
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

        return self
        
    def addRechits(self):
        markerSizeScale = MarkerSizeLogScaler(self.rechits_df_onLayer.rechits_energy, maxMarkerSize=25, minMarkerSize=2)
        #markerSizeScale = MarkerSizeLinearScaler(self.rechits_df_onLayer.rechits_energy, maxMarkerSize=10)
        showLegend = True
        for index, grouped_df in self.rechits_df_onLayer.groupby(by=["clus3D_id", "clus2D_id"], dropna=False):
            # dropna=False to keep outlier rechits and rechits members of outlier layer cluster
            self.fig.add_trace(go.Scatter(
                mode="markers",
                legendgroup="rechits",
                legendgrouptitle_text="Rechits",
                name=f"2D cluster nb {index[1]}",
                x=grouped_df["rechits_x"], y=grouped_df["rechits_y"], 
                marker=dict(
                    symbol="circle",
                    color=self.mapClus2Did_color(index[1]),
                    size=markerSizeScale.scale(grouped_df["rechits_energy"]),
                ),
                customdata=np.dstack((grouped_df.rechits_energy, grouped_df.rechits_rho, grouped_df.rechits_delta,
                    getPointTypeStringForRechits(clus2D_id=index[1], grouped_df=grouped_df)))[0],
                #hovertemplate="clus2D_x=%{x}<br>clus2D_y=%{y}<br>clus2D_z=%{z}<br>clus2D_size=%{marker.size}<extra></extra>",
                hovertemplate=(
                    "Rechit : %{customdata[3]}<br>"
                    "Energy: %{customdata[0]:.2g} GeV<br>Rho: %{customdata[1]:.2g} GeV<br>"
                    "Delta: %{customdata[2]:.2g} cm"
                    #"Size : %{marker.size}"
                )
            ))

            # Plot arrows from follower to nearest higher
            for row in grouped_df.dropna(subset=["rechits_x_ofNearestHigher"]).itertuples(index=False):
                if row.rechits_pointType != 0:
                    # Drop non-followers 
                    continue
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
                        color=self.mapClus2Did_color(index[1]),
                        size=10,
                        angleref="previous",
                        standoff=0.3*markerSizeScale.scaleSingleValue(row.rechits_energy_ofNearestHigher),
                    ),
                    line_width=max(1, math.log(row.rechits_cumulativeEnergy/0.01)), #line width in pixels
                ))
                showLegend = False

        return self

    def addCircleSearchForComputingClusterPosition(self):
        """ Draw a circle around each 2D cluster representing the search area for rechits to compute the 2D cluster position.
        ie a circle centered on the maximum energy cell of the cluster and of radius sqrt(positionDeltaRho2) """
        radius:float = math.sqrt(self.event.clueParameters["positionDeltaRho2"])
        # Find for each 2D cluster the rechit with the highest energy
        for row in self.rechits_df_onLayer.sort_values(["clus2D_id", "rechits_energy"], ascending=False).groupby("clus2D_id").first().itertuples():
            center = np.array([row.rechits_x, row.rechits_y])

            self.fig.add_shape(type="circle", xref="x", yref="y", 
                x0=center[0]-radius, x1=center[0]+radius, y0=center[1]-radius, y1=center[1]+radius)
        return self
    
    def addImpactPoint(self):
        impacts = self.impact_df[self.impact_df.layer == self.layerNb]

        self.fig.add_trace(go.Scatter(
            mode="markers",
            name="Impact from DWC",
            x=impacts.impactX, y=impacts.impactY,
            marker=dict(
                color="black",
                size=8,
                symbol="x"
            ),
            hoverinfo='skip',
        ))
        return self
    
    def addDetectorExtent(self):
        """ Plot detector extent (very approximate detector area) """
        detExt = hists.parameters.DetectorExtentData # For data
        
        self.fig.add_shape(type="circle",
            xref="x", yref="y",
            x0=detExt.centerX-detExt.radius, x1=detExt.centerX+detExt.radius,
            y0=detExt.centerY-detExt.radius, y1=detExt.centerY+detExt.radius,
            line_color="blue",
        )
        return self