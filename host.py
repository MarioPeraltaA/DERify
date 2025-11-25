"""Impact of DER over distribution network.

DER such as:

    - Storage
    - PVSystem
    - WindGen

It is considered inverter-based technollogy.

.. warning::
    ``WindGen`` class interface it is not available in DSS-Extension.

Author::

    Mario Roberto Peralta. A.

"""

# Factory
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from dss import dss, enums, IDSS
import numpy as np
import re
# GIS
import glob
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
# Set to False to leverage multiple contexts
dss.AllowChangeDir = False
dss.AllowForms = False
dss.AllowEditor = False


class Circuit(ABC):
    """Integrated circuit interface."""

    @abstractmethod
    def calculate_losses(self):
        """Compute losses regarding scenario."""
        ...

    @abstractmethod
    def fault_network(self):
        """Run fault study regardomg scenario."""
        ...


@dataclass()
class GISCircuit(ABC):
    """Handle and cope with GIS (No electrical modeling)."""

    gis_path: str = "./GIS/*.shp"
    layers: dict[str, list[gpd.GeoDataFrame, str]] = field(
        default_factory=dict
    )

    def __post_init__(
            self
    ):
        """Retrieve and set layers."""
        self.get_layers()
        self.paint_layers()  # In place

    def read_gis(
        self,
        path: str,
        epsg: int = 5367,
        to_local: bool = False
    ) -> gpd.GeoDataFrame | None:
        """Read a GIS file and return a GeoDataFrame with a consistent CRS.

        By default, the data is returned in EPSG:4326 (WGS84), which is
        required for Folium compatibility. If ``to_local=True``, the data
        is reprojected to the specified local EPSG code, typically used
        for utility-scale electrical modeling (e.g., EPSG:5367 in Costa Rica).

        Parameters
        ----------
        path : str
            File path to the GIS data.
        epsg : int, optional
            EPSG code of the local projection to use. Default is 5367.
        to_local : bool, optional
            If True, the GeoDataFrame is returned in the local projection.
            If False (default), it is returned in EPSG:4326.

        Returns
        -------
        geopandas.GeoDataFrame or None
            GeoDataFrame in the requested CRS, or None if reading fails.

        """
        try:
            gdf = gpd.read_file(path)
        except Exception as e:
            print(f"Error reading GIS file '{path}': {e}")
            return None

        current_epsg = gdf.crs.to_epsg()
        if current_epsg is None:
            logg = (
                f"Warning: No CRS found in '{path}'. "
                "Returning raw GeoDataFrame."
            )
            print(logg)
            return gdf

        target_epsg = epsg if to_local else 4326
        if current_epsg != target_epsg:
            return gdf.to_crs(epsg=target_epsg)
        return gdf

    def get_layers(
            self,
    ):
        """Fill layers up.

        Shapefile name may be: ``<xxxx>_<name>``.

        .. warning::
            Filter out those layers (geodataframes) with no rows.

        """
        shapefiles = glob.glob(self.gis_path)
        shapefiles.sort(reverse=True)
        for shp in shapefiles:
            name = shp.split("/")[-1].split(".")[0].split("_")[-1]
            gdf = self.read_gis(shp)
            try:
                if not gdf.shape[0]:
                    raise ValueError(f"No values in {name}")
            except ValueError as e:
                print(f"EmptyGeoDataFrame: {e}.")
                continue
            else:
                self.layers[name] = [gdf]

    def paint_layers(
            self,
            seed: int = 7859
    ) -> dict[str, list[str, str]]:
        """Assign eye-cathing color to each layer.

        Uses ``rng.shuffle`` instead of ``rng.integers`` to make sure
        all colors are different.

        """
        # Get the list of X11/CSS4 color names
        lib_colors = list(plt.cm.colors.cnames)
        size = len(self.layers)
        # Seed for reproducibility
        rng = np.random.default_rng(seed=seed)
        rnd_ints = np.arange(0, len(lib_colors))
        rng.shuffle(rnd_ints)
        colors = [lib_colors[c] for c in rnd_ints[:size]]
        # Add to dict style in place
        for i, gdf_list in enumerate(self.layers.values()):
            gdf_list.append(colors[i])

    def explore_ckt(
            self
    ) -> folium.Map:
        """Map of the circuit.

        Compute ceneter to start up the map.

        """
        # Get center location
        y_avg = np.average(
            [lay[0].geometry.union_all().centroid.y
             for lay in self.layers.values()]
        )
        x_avg = np.average(
            [lay[0].geometry.union_all().centroid.x
             for lay in self.layers.values()]
        )
        center = [y_avg, x_avg]
        ckt_map = folium.Map(
            crs="EPSG3857",
            zoom_start=15,
            control_scale=True,
            location=center,
            tiles="cartodbpositron"
        )
        # Pile up layers
        for name, layer in self.layers.items():
            gdf, color = layer
            gdf.explore(
                m=ckt_map,
                popup=True,
                tooltip=True,
                name=name,
                color=color,
                show=False
            )

        # Customize tile
        folium.TileLayer("Cartodb dark_matter", show=False).add_to(ckt_map)
        folium.LayerControl().add_to(ckt_map)
        return ckt_map


@dataclass()
class Network(ABC):
    """Abstract Base Class of Circuits (Manufacturer).

    The circuit electrical model.

    There is only one *Head* i.e. **Adj. Degree** of
    feeder is only and solely *one*.

    .. warning::

        Property :py:attr:`dss.Text.Command` when use ``compile``
        modifies the directory's root and sets the model's directory
        as the current one. So it is recommended to run it
        afterwards.

    .. warning::

        Avoid to use command :py:attr:`dss.ActiveCircuit.AllBusNames`
        as buses unique labels may come with *dots* and OpenDSS
        split nodes from bus name at first dot.

    """

    ckt_path: str = "./CKT/CKT_Daily.dss"
    solve_mode: int = enums.SolveModes.Daily
    control_mode: int = enums.ControlModes.Time
    algorithm: int = enums.SolutionAlgorithms.NewtonSolve
    number: int = 96
    stepsize_min: int = 15
    to_solve: bool = True
    feeder_power: np.ndarray[float] | None = None
    der_data = {"Storage": [], "PVSystem": []}
    der_names: list = field(default_factory=list)
    der_dummy_names: list = field(default_factory=list)
    power_monitors:  dict[str, str] = field(default_factory=dict)
    losses_monitors:  dict[str, str] = field(default_factory=dict)
    volt_curr_monitors: dict[str, str] = field(default_factory=dict)
    switches_monitors: list[str] = field(default_factory=list)
    head_monitor: str | None = None   # PWR
    head_meter:  str | None = None
    mv_buses_id: list[str] = field(default_factory=list)
    lv_buses_id: list[str] = field(default_factory=list)
    ckt_losses: np.ndarray[float] | None = None
    ckt_faults: list[
        tuple[dict[str, list[str, np.ndarray, np.ndarray]], float]
    ] = field(default_factory=list)
    dss: IDSS = field(init=False)

    def __post_init__(
            self
    ):
        """Load and solve circuit."""
        self.dss = dss.NewContext()
        self.load_ckt()

    def put_daily_solution_mode(
            self
    ):
        """Type of solution."""
        # Retrieve context interfaces
        dssSolution = self.dss.ActiveCircuit.Solution

        # Set kind of solution
        dssSolution.Mode = self.solve_mode
        dssSolution.ControlMode = self.control_mode
        dssSolution.Number = self.number
        dssSolution.StepsizeMin = self.stepsize_min
        dssSolution.Algorithm = self.algorithm

    def put_fault_solution_mode(
            self
    ):
        """Before running fault simulation.

        Update solution mode and solve after adding fault.

        ..warning::
            Inverter-Based techonology i.e. DER devices,
            are turned off in order to run Dynamic simulation.

        """
        self.dss.ActiveCircuit.Solution.Mode = enums.SolveModes.Dynamic
        self.dss.ActiveCircuit.Solution.Number = 1
        self.dss.ActiveCircuit.Solution.StepSize = 1  # sec

    def load_ckt(
            self
    ):
        """Load and solve network."""
        # Compile a model
        self.dss.Text.Command = f'compile "{self.ckt_path}"'
        self.put_daily_solution_mode()
        if self.to_solve:
            self.dss.ActiveCircuit.Solution.Solve()

    def add_monitor(
            self,
            full_name_element: str = "transformer.substation",
            monitor_id: str = "substation_monitor_1",
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power,
            polar: bool = False
    ) -> str:
        """Instantiate and set a single monitor."""
        dssMonitors = self.dss.ActiveCircuit.Monitors
        monitors_id = dssMonitors.AllNames
        if monitor_id in monitors_id:
            return monitor_id

        if mode == enums.MonitorModes.VI:
            notation = "VIpolar"
        else:
            notation = "ppolar"
        self.dss.Text.Command = f"new monitor.{monitor_id} {notation}={polar}"
        dssMonitors.Name = monitor_id
        dssMonitors.Element = full_name_element
        dssMonitors.Terminal = terminal
        dssMonitors.Mode = mode
        return monitor_id

    def add_meter(
            self,
            full_name_element: str = "transformer.substation",
            meter_id: str = "substation_meter",
            terminal: int = 1
    ) -> str:
        """Instantiate and set a single EnergyMeter."""
        self.dss.Text.Command = (
            f"New EnergyMeter.{meter_id} "
            f"element={full_name_element} "
            f"terminal={terminal}"
        )
        return meter_id

    def add_head_monitor(
            self,
            source_bus_id: str = "sourcebus",
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power
    ):
        """Deploy monitors to each PDE connected to sourcebus.

        To keep an eye on external network power.

        """
        ibus_obj = self.dss.ActiveCircuit.ActiveBus(source_bus_id)
        pd_elements = ibus_obj.AllPDEatBus
        # Full name branches
        pd_elements = self.dss.ActiveCircuit.ActiveBus.AllPDEatBus
        pd_elements = [
            None if (e) and (e.lower() in {"none", "nan", "null"}) else e
            for e in pd_elements
        ]
        # Kick out falsy items
        feeder_branches = list(filter(None, pd_elements))
        # Add monitor
        try:
            if feeder_branches:
                n_branches = len(feeder_branches)
                if n_branches != 1:
                    raise TypeError("Multiple branches at head.")
            else:
                raise ValueError("No PDE at feeder.")
        except ValueError as e:
            logg = (
                f"EmptyBranches: {e}"
            )
            print(logg)
        except TypeError as e:
            logg = (
                f"NonUniqueHead: {e}"
            )
        else:
            branch = feeder_branches[0]
            _ = self.dss.ActiveCircuit.SetActiveElement(branch)
            element_id = self.dss.ActiveClass.Name
            monitor_id = self.add_monitor(
                branch, f"{element_id}_monitor_{mode}", terminal, mode
            )
            self.head_monitor = monitor_id

    def add_head_meter(
            self,
            source_bus_id: str = "sourcebus",
            terminal: int = 1,
    ):
        """Embed EnergyMeter right at feeders head.

        To assess Topology analysis and collect global Registers.

        """
        ibus_obj = self.dss.ActiveCircuit.ActiveBus(source_bus_id)
        pd_elements = ibus_obj.AllPDEatBus
        # Full name branches
        pd_elements = self.dss.ActiveCircuit.ActiveBus.AllPDEatBus
        pd_elements = [
            None if (e) and (e.lower() in {"none", "nan", "null"}) else e
            for e in pd_elements
        ]
        # Kick out falsy items
        feeder_branches = list(filter(None, pd_elements))
        # Add meter
        try:
            if feeder_branches:
                n_branches = len(feeder_branches)
                if n_branches != 1:
                    raise TypeError("Multiple branches at head.")
            else:
                raise ValueError("No PDE at feeder.")
        except ValueError as e:
            logg = (
                f"EmptyBranches: {e}"
            )
            print(logg)
        except TypeError as e:
            logg = (
                f"NonUniqueHead: {e}"
            )
        else:
            branch = feeder_branches[0]
            _ = self.dss.ActiveCircuit.SetActiveElement(branch)
            element_id = self.dss.ActiveClass.Name
            meter_id = self.add_meter(
                branch, f"{element_id}_meter", terminal
            )
            self.head_meter = meter_id

    def get_monitor_data(
            self,
            monitor_id: str = "feeder_pq",
            reset: bool = True
    ) -> np.ndarray:
        """Key and retrieve monitor's data.

        Active Circuit must be run already.

        """
        dssMonitors = self.dss.ActiveCircuit.Monitors
        # Activete monitor element
        dssMonitors.Name = monitor_id

        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                if dssMonitors.Name == monitor_id:
                    # Retrieve data
                    monitor_data = dssMonitors.AsMatrix()
                    if reset:
                        dssMonitors.Reset()   # Reset only active one
                else:
                    raise ValueError(f"Monitor {monitor_id} not found")

            else:
                raise RuntimeError(
                    f"Circuit {self.dss.ActiveCircuit.Name} did not converge"
                )
        except RuntimeError as e:
            print(f"MaxIterReached: {e}.")
            return
        except ValueError as e:
            print(f"ElementNotFound: {e}.")
            return

        return monitor_data

    def get_meter_data(
            self,
            meter_id: str = "substation_meter",
            register_i: int = enums.EnergyMeterRegisters.kWh
    ) -> float:
        """Retrieve requested Register value from EnergyMeter.

        .. warning::
            EnergyMeter Registers are neither clear up
            nor reset after getting its data.

        """
        dssMeters = self.dss.ActiveCircuit.Meters
        dssMeters.Name = meter_id
        return dssMeters.RegisterValues[register_i]

    def catch_der(
            self,
            class_name: str = "Storage",
            full_name: str = "Storage.mv_battery"
    ):
        """Add DER to internall data.

        Capture properties to make up temporary artifitial
        dummy generator for the sake of short circuit faults.

        """
        dssElement = self.dss.ActiveCircuit.ActiveElement(full_name)

        der_name = full_name.split(".")[-1]
        full_dummy_name = f"Generator.dummy_{der_name}"
        if full_dummy_name in self.der_dummy_names:
            return
        bus_nodes = dssElement.Properties("Bus1").Val
        phases = dssElement.Properties("Phases").Val
        kvoltage = dssElement.Properties("kV").Val
        connection = dssElement.Properties("Conn").Val
        kva_power = dssElement.Properties("kVA").Val
        dss_command = (
            f"New {full_dummy_name} "
            f"Bus1={bus_nodes} "
            f"Phases={phases} "
            f"kV={kvoltage} "
            f"Conn={connection} "
            f"kVA={kva_power} "
            f"model=3"
        )
        self.der_names.append(full_name)
        self.der_dummy_names.append(full_dummy_name)
        self.der_data[class_name].append(
            (full_name, full_dummy_name, dss_command)
        )

    def embed_dummy_generation(
            self,
            enabled: bool = False
    ):
        """Integrate generation that mimics DER."""
        for data in self.der_data.values():
            for _, dummy_id, generator in data:
                self.dss.Text.Command = generator
                if not enabled:
                    self.dss.Text.Command = f"Disable {dummy_id}"

    def flip_der(
            self
    ):
        """Turn on and off DER devices."""
        dssCircuit = self.dss.ActiveCircuit
        der_names = self.der_names
        dummy_names = self.der_dummy_names

        in_circuit = []
        for full_name in der_names:
            dssElement = dssCircuit.ActiveElement(full_name)
            in_circuit.append(dssElement.Enabled)

        # Turn DER off
        if all(in_circuit):
            for der, dummy in zip(der_names, dummy_names):
                dssCircuit.Disable(der)
                dssCircuit.Enable(dummy)

        # Turn DER on
        elif not any(in_circuit):
            for der, dummy in zip(der_names, dummy_names):
                dssCircuit.Enable(der)
                dssCircuit.Disable(dummy)

        # Update circuit
        self.dss.ActiveCircuit.Solution.Solve()

    def deploy_pce_monitors(
            self,
            terminals: tuple[int] = (1, 1, 1),
            modes: tuple[int] = (
                enums.MonitorModes.Power,   # P-Q
                enums.MonitorModes.VI,      # Volt-Curr
                9                           # Losses
            )
    ):
        """Connect measurement infrastructure to PCE.

        Power Convertion Elements (PCE) regarding
        local network in order to measure losses. Bear in mind
        that default ``VSource`` elements it is not considered a PCE.

        Classify either Power, VI or Losses kind of
        monitors and its representation is rectangular
        with Real and Imaginary part.

        .. Warning::
            Ensure to call this method in the proper
            monitoring kind of mode.

        .. Note::
            DER devices full names are internally retained
            by keys: ``Storage`` ``PVSystem``.

        """
        # PCE of local network
        i = self.dss.ActiveCircuit.FirstPCElement()
        if i:
            while i:
                full_name = self.dss.ActiveCircuit.ActiveElement.Name
                _ = self.dss.ActiveCircuit.ActiveElement(full_name)
                class_name = self.dss.ActiveClass.ActiveClassName
                if class_name in self.der_data:
                    self.catch_der(class_name, full_name)
                element_id = self.dss.ActiveCircuit.ActiveClass.Name
                for terminal, mode in zip(terminals, modes):
                    monitor_id = self.add_monitor(
                        full_name,
                        f"{element_id}_monitor_{mode}",
                        terminal,
                        mode,
                        polar=False
                    )
                    if mode == enums.MonitorModes.Power:
                        self.power_monitors[full_name.lower()] = monitor_id
                    elif mode == enums.MonitorModes.VI:
                        self.volt_curr_monitors[full_name.lower()] = monitor_id
                    elif mode == 9:
                        self.losses_monitors[full_name.lower()] = monitor_id

                i = self.dss.ActiveCircuit.NextPCElement()

    def deploy_switches_monitors(
            self,
            polar: bool = True
    ):
        """Embed voltage-current monitor to SwitchedObj."""
        dssSwitch = self.dss.ActiveCircuit.SwtControls
        switched_objs: list[tuple] = []
        i = dssSwitch.First
        while i:
            switched_objs.append(
                (dssSwitch.Name, dssSwitch.SwitchedObj)
            )
            i = dssSwitch.Next
        if not switched_objs:
            return
        else:
            for switch_id, full_name in switched_objs:
                switch_number = switch_id.split("_")[-1]   # Check name
                monitor_id = self.add_monitor(
                    full_name_element=full_name,
                    monitor_id=f"monitor_{switch_number}_vi",
                    terminal=1,
                    mode=enums.MonitorModes.VI,
                    polar=polar
                )
                self.switches_monitors.append(monitor_id)

    def external_network_power(
            self,
    ) -> np.ndarray[float]:
        """Power flowing into domestic network.

        Retrieve data from those monitors connected
        to external network brances (PDE).
        *i.e.*, Elements that work as bridge between external
        and domestic network. *e.g.* Substation.

        """
        injected_power = np.zeros(
            (self.dss.ActiveCircuit.Solution.Number, 2)
        )
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                data = self.get_monitor_data(self.head_monitor)
                injected_power[:, 0] += data[:, 2::2].sum(axis=1)
                injected_power[:, 1] += data[:, 3::2].sum(axis=1)
            else:
                raise RuntimeError(
                    f"Circuit {self.dss.ActiveCircuit.Name} did not converge"
                )
        except RuntimeError as e:
            print(f"MaxIterReached: {e}.")
            return
        else:
            self.feeder_power = injected_power
            return injected_power

    def local_mismatch(
            self,
    ) -> np.ndarray[float]:
        """Compute mismatch generation-demand.

        Add up each PCE all along the timeframe from
        monitors data. If monitors were set as mode ``9``
        then it measures losses, so that, the next
        convention if followed:

            - If negative active power, then the element generates
            real power [kW].
            - If positive reactive power, then the element
            absorbs reactive [kVAr].

        Returns
        -------
        delta_matrix : np.ndarray[float]
            Switched sign so positive remaining either
            actitive or reactive it is seen as a surplus.

        .. Note::
            Without external network contribution, so far.

        """
        delta_matrix = np.zeros(
            (self.dss.ActiveCircuit.Solution.Number, 2)
        )
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                for name in self.losses_monitors.values():
                    data = self.get_monitor_data(name)
                    delta_matrix[:, 0] += data[:, 2] / 1e3  # kW
                    delta_matrix[:, 1] += data[:, 3] / 1e3  # kVAr
            else:
                raise RuntimeError(
                    f"Circuit {self.dss.ActiveCircuit.Name} did not converge"
                )
        except RuntimeError as e:
            print(f"MaxIterReached: {e}.")
            return
        else:
            return -delta_matrix

    def get_switched_profiles(
            self
    ) -> dict[str, np.ndarray]:
        """Get VI daily data from switched objects.

        These may be critical lines or bridges.

        """
        data: dict[str, np.ndarray] = {}
        for monitor_id in self.switches_monitors:
            measures = self.get_monitor_data(monitor_id)
            data[monitor_id] = measures
        return data

    def get_vi_profiles(
            self
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get VI daily data from Power Conversion Elements (PCE).

        Retrain hot phases only.

        Returns
        -------
        data : dict[str, tuple[np.ndarray, np.ndarray]]
            Dictionary whose key is the full name PCE element
            and its values tuples of phase voltage and current
            respectively as complex datatype.

        .. Note::

            Voltage phase values are returned as p.u. based on its
            ``kV`` asigned property which it is supossed to be
            line to line nominal voltage.

        """
        data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for at_element, monitor_id in self.volt_curr_monitors.items():
            elem = self.dss.ActiveCircuit.ActiveCktElement(at_element)
            fts = elem.Properties
            base_v = float(fts("kV").Val) * 1e3   # kV to V
            vi_data = self.get_monitor_data(monitor_id)
            m = 2
            n = m + len(elem.Voltages)
            v = vi_data[:, m:n]  # Voltages
            c = vi_data[:, n:]   # Currents
            # Turn into complex data type per entry
            v_phasors = v.reshape(len(v), -1, 2).dot([1, 1j])
            c_phasors = c.reshape(len(c), -1, 2).dot([1, 1j])
            # Drop dead phases
            v_hot_phases = ~np.all(v_phasors == 0 + 0j, axis=0)
            c_hot_phases = ~np.all(c_phasors == 0 + 0j, axis=0)
            # Update phasors
            v_phasors = v_phasors[:, v_hot_phases] * 1 / base_v   # pu
            c_phasors = c_phasors[:, c_hot_phases]
            data[at_element] = (v_phasors, c_phasors)
        return data

    def three_phase_fault(
            self,
            fault_id: str = "LLLG_busX",
            bus_id: str = "busX",
            terminals: list[str] = [".1", ".2", ".3"],
            n_phases: int = 3,
            resistance: float = 1e-2
    ):
        """Set LLLG type of fault."""
        from_nodes = "".join(terminals)
        to_nodes = ".0.0.0"
        from_bus = f"{bus_id}{from_nodes}"
        to_bus = f"{bus_id}{to_nodes}"

        self.dss.Text.Command = (
            f"edit Fault.{fault_id} "
            f"bus1={from_bus} bus2={to_bus} "
            f"phases={n_phases} r={resistance}"
        )

    def single_phase_fault(
            self,
            fault_id: str = "LG_busX",
            bus_id: str = "busX",
            terminals: list[str] = [".1"],
            n_phases: int = 1,
            resistance: float = 1e-2
    ):
        """Set LG type of fault."""
        from_nodes = terminals[0]
        to_nodes = ".0"
        from_bus = f"{bus_id}{from_nodes}"
        to_bus = f"{bus_id}{to_nodes}"

        self.dss.Text.Command = (
            f"edit Fault.{fault_id} "
            f"bus1={from_bus} bus2={to_bus} "
            f"phases={n_phases} r={resistance}"
        )

    def double_line_fault_a(
            self,
            fault_id: str = "LLG_a_busX",
            bus_id: str = "busX",
            terminals: list[str] = [".1", ".2"],
            n_phases: int = 2,
            resistance: float = 1e-2
    ):
        """Set LLG_a type of fault."""
        from_nodes = f"{terminals[0]}{terminals[0]}"
        to_nodes = f"{terminals[1]}.0"
        from_bus = f"{bus_id}{from_nodes}"
        to_bus = f"{bus_id}{to_nodes}"

        self.dss.Text.Command = (
            f"edit Fault.{fault_id} "
            f"bus1={from_bus} bus2={to_bus} "
            f"phases={n_phases} r={resistance}"
        )

    def double_line_fault_b(
            self,
            fault_id: str = "LLG_b_busX",
            bus_id: str = "busX",
            terminals: list[str] = [".1", ".2"],
            n_phases: int = 2,
            resistance: float = 1e-2
    ):
        """Set LLG_b type of fault."""
        from_nodes = "".join(terminals)
        to_nodes = ".0.0"
        from_bus = f"{bus_id}{from_nodes}"
        to_bus = f"{bus_id}{to_nodes}"

        self.dss.Text.Command = (
            f"edit Fault.{fault_id} "
            f"bus1={from_bus} bus2={to_bus} "
            f"phases={n_phases} r={resistance}"
        )

    def float_line_line_fault(
            self,
            fault_id: str = "LL_busX",
            bus_id: str = "busX",
            terminals: list[str] = [".1", ".2"],
            n_phases: int = 1,
            resistance: float = 1e-2
    ):
        """Set LL type of fault."""
        from_nodes = terminals[0]
        to_nodes = terminals[1]
        from_bus = f"{bus_id}{from_nodes}"
        to_bus = f"{bus_id}{to_nodes}"

        self.dss.Text.Command = (
            f"edit Fault.{fault_id} "
            f"bus1={from_bus} bus2={to_bus} "
            f"phases={n_phases} r={resistance}"
        )

    def set_fault(
            self,
            bus_id: str = "busX",
            fault_type: str = "LLLG",
            terminals: list[str] = [".1", ".2", ".3"]
    ) -> str:
        """Define and model a fault element.

        .. Note::
            Number of nodes of bus must be equal
            or greater than number of phases of
            the requested fault.

        """
        fault_handlers = {
            "LLLG": self.three_phase_fault,
            "LG": self.single_phase_fault,
            "LLG_a": self.double_line_fault_a,
            "LLG_b": self.double_line_fault_b,
            "LL": self.float_line_line_fault
        }
        try:
            if fault_type in fault_handlers:
                fault_id = f"{fault_type}_{bus_id}"
                self.dss.Text.Command = (
                    f"new Fault.{fault_id}"
                )
                fault_handlers[fault_type](
                    fault_id, bus_id, terminals
                )
            else:
                raise KeyError("Unknown fault type")
        except KeyError as e:
            print(f"NoFaultType: {e}.")
            return
        else:
            return fault_id

    def get_fault_data(
            self,
            busx_id: str,
            fault_type: str
    ) -> tuple[dict, float]:
        """Retrieve and return fault data at certain bus.

        It retains magnitude of fault current only and due to
        unbalance network it gets the highest phase current measured.

        It slices half the array because one terminal is enough.

        .. warning::

            Command :py:attr:`dss.ActiveCircuit.ActiveBus.AllPDEatBus`
            it is not reliable as may return non neighbor branches.
            You may use :py:class:`gdss.CktGraph` to walk around.

        """
        fault_data: dict[str, list[str, float, np.ndarray]] = {}
        # Retrieve data during fault
        dssBus = self.dss.ActiveCircuit.ActiveBus(busx_id)
        bus_voltage = dssBus.VMagAngle   # VLN-Magnitude [V], angle [deg]
        bus_distance = dssBus.Distance   # [km]
        Isc = 0.0                        # [A]
        for branch in dssBus.AllPDEatBus:
            if branch:
                dssBranch = self.dss.ActiveCircuit.ActiveElement(branch)
                currents = dssBranch.CurrentsMagAng
                current_phasor = currents[:len(currents)//2]
                phase_current_mag = current_phasor[::2]
                # Add up most severe phase current magnitude.
                Isc += max(phase_current_mag)
        fault_data[fault_type] = [
            busx_id,
            Isc,
            bus_voltage
        ]
        return fault_data, bus_distance

    def run_fault_study(
            self,
            **kwargs
    ):
        """Set solution context to run fault studies.

        Make sure :math:`Z^{(1)}` and :math:`Z^{(0)}`, due to
        external network, were properly set in the current circuit.

        Parameters
        ----------
        kwargs : dict[str, list[str]]
            Bus id to be faulted and the type of faults
            to be addressed.

            - LLLG: Three phase to ground.
            - LG: Single phase to ground.
            - LLG_a: Double line to ground.
            - LLG_b: Each two line to ground.
            - LL: Line to Line (essentially a single phase fault).

        .. Note::
            Fault gets disabled after its data is
            collected and set it as circuit attribute.

        """
        dssCircuit = self.dss.ActiveCircuit
        try:
            if dssCircuit.Solution.Converged:
                self.embed_dummy_generation(enabled=False)
                self.flip_der()  # Turn DER off
                # Fault environment
                self.put_fault_solution_mode()
            else:
                raise RuntimeError(
                    f"Circuit {dssCircuit.Name} did not converge"
                )
        except RuntimeError as e:
            print(f"MaxIterReached: {e}.")
        else:
            for busx, (fault_types, terminals) in kwargs.items():
                for fault_type in fault_types:
                    # Set fault
                    fault_id = self.set_fault(
                        busx, fault_type, terminals
                    )
                    self.dss.ActiveCircuit.Solution.Solve()
                    # Remove fault
                    dssCircuit.Disable(fault_id)
                    # Retrieve data
                    fault_data, distance = self.get_fault_data(
                        busx_id=busx,
                        fault_type=fault_type
                    )
                    # Store data
                    self.ckt_faults.append((fault_data, distance))
        finally:
            self.ckt_faults.sort(key=lambda x: x[1])   # Sort by distance
            # Set back active circuit to last steady state
            dssCircuit.Solution.Cleanup()
            self.put_daily_solution_mode()
            self.flip_der()    # Put back DER

    def get_fault_currents(
            self,
    ) -> tuple[list[np.ndarray[float, float]], list[list[str]]]:
        """Filter short circuit phase current magnitude."""
        # LLL-G: Three phase bolted Fault
        three_phase_fault = []
        # L-G: Single Line-to-Ground Fault
        single_phase_fault = []
        # LLG_a: Double Line-to-ground Fault
        complex_double_fault = []
        # LLG_b: Each Line-to-ground Fault
        simple_double_fault = []
        # LL: Line-to-Line Fault
        float_double_fault = []
        # Catch Magnitude only
        for faults, distance in self.ckt_faults:
            if "LLLG" in faults:
                three_phase_fault.append(
                    (faults['LLLG'][1],
                     distance, faults['LLLG'][0])
                )

            if "LG" in faults:
                single_phase_fault.append(
                    (faults['LG'][1],
                     distance, faults['LG'][0])
                )

            if "LLG_a" in faults:
                complex_double_fault.append(
                    (faults['LLG_a'][1],
                     distance, faults['LLG_a'][0])
                )

            if "LLG_b" in faults:
                simple_double_fault.append(
                    (faults['LLG_b'][1],
                     distance, faults['LLG_b'][0])
                )

            if "LL" in faults:
                float_double_fault.append(
                    (faults['LL'][1],
                     distance, faults['LL'][0])
                )
        # Vectorize: (Isc [A], distance [km])
        fault_data: list[np.ndarray[float, float]] = []
        fault_buses: list[list[str]] = []
        for fault in [
            three_phase_fault,
            single_phase_fault,
            complex_double_fault,
            simple_double_fault,
            float_double_fault
        ]:
            fault_matrix = np.empty(
                (len(fault), 2)
            )
            currents = [c[0] for c in fault]
            distances = [d[1] for d in fault]
            buses_id = [b[2] for b in fault]
            fault_matrix[:, 0] = currents
            fault_matrix[:, 1] = distances
            fault_data.append(fault_matrix)
            fault_buses.append(buses_id)

        return fault_data, fault_buses


@dataclass()
class CktGraph(ABC):
    """Skeleton factory."""

    ckt: Network
    vertices: list[str] = field(default_factory=list)
    edges: list[tuple[str, str]] = field(default_factory=list)
    adj: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Set graph given the Network."""
        self.set_adjacency_list()

    def add_vertex(
        self,
        vertex_id: str
    ) -> str:
        """Instantiate vertex.

        Vertex is a unique bus of the circuit in spite
        of its nodes.

        .. Note::
            In opendss a Bus may have multiple Nodes.

        """
        if vertex_id not in self.vertices:
            self.vertices.append(vertex_id)

    def add_edge(
        self,
        from_vertex: str,
        to_vertex: str
    ):
        """Instantiate edge.

        Edge is a branch with two ends. i.e. Connection
        between two vertices.

        """
        self.add_vertex(from_vertex)
        self.add_vertex(to_vertex)
        edge = (from_vertex, to_vertex)
        if edge not in self.edges:
            self.edges.append(edge)

    def untwist_branch(
            self,
            buses: list[str],
    ):
        """Cope with odd branches.

        Odd branches (more than two ends/terminals) such
        as three phase three winding transformer are taken
        as a cycle graph third order :math:`C_{3}`.

        """
        branches = [
            (buses[i], buses[(i + 1) % len(buses)]) for i in range(len(buses))
        ]
        for edge in branches:
            if "hvmv_3" in edge:
                continue
            self.add_edge(edge[0], edge[1])

    def collect_branches(
            self
    ):
        """Gather branches in all zones seen by Meters."""
        dssMeters = self.ckt.dss.ActiveCircuit.Meters
        i = dssMeters.First
        branches: list[str] = []
        while i:
            branches += dssMeters.AllBranchesInZone
            i = dssMeters.Next
        return branches

    def build_graph(
            self
    ):
        """Generate undirected graph."""
        branches = self.collect_branches()
        dssCircuit = self.ckt.dss.ActiveCircuit
        for edge in branches:
            if edge:
                dssBranch = dssCircuit.ActiveCktElement(edge)
                nodes = dssBranch.BusNames
                # Strip specific nodes
                buses = [re.sub(r'(\.\d+)+$', '', node) for node in nodes]
                if len(set(buses)) != 2:
                    self.untwist_branch(buses)
                else:
                    self.add_edge(buses[0], buses[1])

    def set_adjacency_list(
            self
    ):
        """Graph representation."""
        self.build_graph()
        self.adj = {
            v: [] for v in self.vertices
        }
        for edge in self.edges:
            self.adj[edge[0]].append(edge[1])
            self.adj[edge[1]].append(edge[0])

    def dfs_edges(
            self,
            graph_adj: dict[str, list[str]],
            root_bus: str | None = "sourcebus",
            depth_limit: int | None = None
    ):
        """Depth First Search.

        To traverse graph. If ``root`` (source) is provided
        then yield only edges in the component reachable
        from source. This pattern mimics `networkX <https://networkx.org/>`_.
        See [1]_ and [2]_.

        References
        ----------
        .. [1] http://www.ics.uci.edu/~eppstein/PADS
        .. [2] https://en.wikipedia.org/wiki/Depth-limited_search

        .. note::

            The ``root`` is not necessary the *bus head* of the
            electrical network.

        """
        if root_bus is None:
            # Edges for all components
            vertices = list(graph_adj.keys())
        else:
            # Edges for components with source
            vertices = [root_bus]

        if depth_limit is None:
            depth_limit = len(graph_adj)

        visited = set()
        for start in vertices:
            if start in visited:
                continue
            visited.add(start)
            stack = [(start, graph_adj[start])]
            depth_now = 1
            while stack:
                parent, children = stack[-1]
                for child in children:
                    if child not in visited:
                        # Discovered edge
                        yield parent, child
                        visited.add(child)
                        if depth_now < depth_limit:
                            # Add child and grandchildren to stack
                            stack.append((child, graph_adj[child]))
                            depth_now += 1
                            break
                else:
                    _ = stack.pop()
                    depth_now -= 1

    def is_connected(
            self,
            graph_adj: dict[str, list[str]]
    ) -> bool:
        """Verify if graph is connected."""
        vertices: list[str] = list(graph_adj.keys())
        edges = self.dfs_edges(
            graph_adj=graph_adj,
            root_bus=vertices[0],
            depth_limit=None
        )
        sub_vertices = {end for ends in edges for end in ends}
        return len(sub_vertices) == len(vertices)


@dataclass()
class BaseCicuit(Network, Circuit):
    """Current circuit."""

    ckt_graph: CktGraph = field(init=False)

    def __post_init__(self):
        """Call Network and set base scenario."""
        super().__post_init__()
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                self.add_head_meter()
                self.add_head_monitor()           # Power monitor
                self.deploy_pce_monitors()        # PQ, VI, Losses monitors
                self.deploy_switches_monitors(polar=True)   # VI monitors
                self.dss.ActiveCircuit.Solution.Solve()
            else:
                raise RuntimeError("Circuit must be initialized")
        except RuntimeError as e:
            print(f"NonSolvedCkt: {e}.")
        else:
            self.ckt_graph = self.make_graph()

    def make_graph(
            self
    ) -> CktGraph:
        """Set graph of current circuit scenario.

        Factory for ``ckt_graph``.
        Override if different graph class is needed.

        """
        return CktGraph(self)

    def calculate_losses(
            self
    ) -> np.ndarray:
        """Return and set global network losses."""
        delta_matrix = self.local_mismatch()
        external_gen = self.external_network_power()
        delta_matrix += external_gen
        self.ckt_losses = delta_matrix
        return delta_matrix

    def fault_network(
            self
    ):
        """Run fault study of short circuit all across the circuit.

        Filter out fault types regarding the number of nodes
        of bus. Finally add up DER short current contribution.

        Classify MV buses and LV ones.

        .. Note::
            Above voltage magnitude of 30kV LL it is considered
            medium voltage.

        .. warning::
            Both ``Vsourcebus`` and renamed third
            floating winding ``hvmv_3`` are skiped.

        """
        fault_buses = {}
        for bus_id in self.dss.ActiveCircuit.AllBusNames:
            dssBus = self.dss.ActiveCircuit.ActiveBus(bus_id)
            skip_bus = (
                dssBus.Name not in self.ckt_graph.vertices,
                bus_id.lower() in {"sourcebus", "hvmv_3"}
            )
            if any(skip_bus):
                continue
            nodes = dssBus.Nodes
            voltages = dssBus.VMagAngle  # mag VLN [V], phase [Deg]
            if all(voltages[::2] >= 30.0e3/np.sqrt(3)):
                self.mv_buses_id.append(bus_id)
            else:
                self.lv_buses_id.append(bus_id)

            terminals = [f".{n}" for n in nodes if n]
            if len(nodes) == 1:
                fault_buses[bus_id] = (['LG'], terminals)
            else:
                fault_buses[bus_id] = (
                    ['LG', 'LLG_a', 'LLG_b', 'LL'], terminals
                )
                if len(nodes) >= 3:
                    fault_buses[bus_id][0].append('LLLG')

        self.run_fault_study(**fault_buses)
        return self.ckt_faults


@dataclass()
class DERCircuit(Network, Circuit):
    """DER-augmented circuit."""

    bess_attrs: list[dict] = field(default_factory=list)
    pvsys_attrs: list[dict] = field(default_factory=list)
    storages_id: list[str] = field(default_factory=list)
    pvsystems_id: list[str] = field(default_factory=list)
    volt_curr_der_monitors: list[str] = field(default_factory=list)
    ckt_graph: CktGraph = field(init=False)

    def __post_init__(self):
        """Call Network, add DER and then monitors."""
        super().__post_init__()
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                # -- DER
                self.add_bess()
                self.add_pv_systems()
                # -- Measuring
                self.add_head_meter()
                self.add_head_monitor()           # Power monitor
                self.deploy_pce_monitors()        # PQ, VI, Losses monitors
                self.deploy_switches_monitors(polar=True)   # VI monitors
                self.deploy_bess_monitors(polar=True)       # VI monitors
                self.deploy_pv_monitors(polar=True)         # VI monitors
                self.dss.ActiveCircuit.Solution.Solve()
            else:
                message: str = (
                    "Circuit must be initialized "
                    "and solved before adding DER"
                )
                raise RuntimeError(message)
        except RuntimeError as e:
            print(f"NonSolvedCkt: {e}.")
        else:
            self.ckt_graph = self.make_graph()

    def make_graph(
            self
    ) -> CktGraph:
        """Set graph of current circuit scenario.

        Factory for ``ckt_graph``.
        Override if different graph class is needed.

        """
        return CktGraph(self)

    def deploy_bess_monitors(
            self,
            polar: bool = True
    ):
        """Embed voltage-current monitor to BESS."""
        dssStorage = self.dss.ActiveCircuit.Storages
        storage_objs: list[str] = dssStorage.AllNames
        storage_objs = [
            None if (e.lower() in {"none", "nan", "null"}) else e
            for e in storage_objs if e
        ]
        storage_objs = list(filter(None, storage_objs))
        for id in storage_objs or []:
            monitor_id = self.add_monitor(
                full_name_element=f"Storage.{id}",
                monitor_id=f"bess_monitor_{id}_vi",
                terminal=1,
                mode=enums.MonitorModes.VI,
                polar=polar
            )
            self.volt_curr_der_monitors.append(monitor_id)

    def set_bess_dispatch_curve(
            self,
            dispatch_curve_id: str = "dispatch_shape",
            npts: int = 96,
            minterval: int = 15,
            hours_soc: tuple[
                tuple[float, float], tuple[float, float]
            ] | None = ((3, 6), (20, 27)),
            charge_pace: float = 1.0,
            discharge_pace: float = 1.0,
            pmult_curve: np.ndarray[float] | None = None
    ):
        """Define dynamically LoadShape.

        Generic daily dispatch shape curve of storage device.

        Parameters
        ----------
        hours_soc : tuple[tuple, tuple]
            Time boundaries for the State of Charge (SoC) of
            storage device where the outer tuple defines
            if charge or discharge and the inner
            set the hours from and to as **24-hr** fashion.

        """

        def hour_to_index(hour: float) -> int:
            """Convert hour (may overflow 24) to loadshape index."""
            hour_wrapped = hour % 24.0
            return int((hour_wrapped / 24.0) * npts)

        def apply_range(
                arr: np.ndarray,
                start_h: float,
                end_h: float,
                pace_value: float
        ):
            """Apply pace value on array handling wrap-around."""
            i = hour_to_index(start_h)
            j = hour_to_index(end_h)

            if start_h % 24 <= end_h % 24:
                # Normal non-wrap case: e.g. 1 → 5
                arr[i:j] = pace_value
            else:
                # Wrap-around: e.g. 23 to 28 (where 28→4)
                arr[i:] = pace_value      # from i to end of day
                arr[:j] = pace_value      # from start to j

        dssLoadShape = self.dss.ActiveCircuit.LoadShapes
        dssLoadShape.New(dispatch_curve_id)
        dssLoadShape.Name = dispatch_curve_id
        dssLoadShape.Npts = npts
        dssLoadShape.MinInterval = minterval
        daily_dispatch = np.zeros(npts)

        if pmult_curve is not None:
            dssLoadShape.UseActual = True
            dssLoadShape.Pmult = pmult_curve
        else:
            dssLoadShape.UseActual = False

            # Apply charge (negative)
            charge_from, charge_to = hours_soc[0]
            apply_range(
                daily_dispatch, charge_from, charge_to, -charge_pace
            )

            # Apply discharge (positive)
            dis_from, dis_to = hours_soc[1]
            apply_range(
                daily_dispatch, dis_from, dis_to, discharge_pace
            )

            dssLoadShape.Pmult = daily_dispatch

    def set_battery(
            self,
            daily_id: str = "dispatch_shape",
            storage_id: str = "mv_battery",
            bus_id: str = "busMV3",
            phases: int = 3,
            kV: float = 34.5,
            kW: float = 10.0,
            kWh: float = 40.0,
            model: int = 1,
            per_stored: float = 10.0,
            per_reserve: float = 10.0,
            dispatch_mode: str = "follow",
            per_efficiencies: tuple[float] = (95.0, 95.0),
            dispatch_schedule: tuple[
                tuple[float, float], tuple[float, float]
            ] | None = None,
            charge_pace: float = 1.0,
            discharge_pace: float = 1.0,
            dispatch_curve: np.ndarray[float] | None = None
    ):
        """Integrate BESS to the circuit."""
        if dispatch_schedule is not None:
            self.set_bess_dispatch_curve(
                dispatch_curve_id=daily_id,
                hours_soc=dispatch_schedule,
                charge_pace=charge_pace,
                discharge_pace=discharge_pace,
                dispatch_curve=None
            )
        elif dispatch_curve is not None:
            self.set_bess_dispatch_curve(
                dispatch_curve_id=daily_id,
                hours_soc=None,
                pmult_curve=dispatch_curve
            )
        self.dss.Text.Command = (
            f"New Storage.{storage_id} phases={phases} "
            f"bus1={bus_id} kV={kV} "
            f"kWrated={kW} kWhrated={kWh} %stored={per_stored} "
            f"%reserve={per_reserve} "
            f"%effcharge={per_efficiencies[0]} "
            f"%effdischarge={per_efficiencies[1]} "
            f"dispmode={dispatch_mode} "
            f"model={model} daily={daily_id}"
        )

        try:
            dssStorages = self.dss.ActiveCircuit.Storages
            dssStorages.Name = storage_id
            if dssStorages.Name == storage_id:
                dssStorages.State = enums.StorageStates.Idling
            else:
                raise ValueError(
                    f"Storage {storage_id} was "
                    "neither added nor activated"
                )
        except ValueError as e:
            print(f"ElementNotFound: {e}.")
            return
        else:
            self.storages_id.append(storage_id)

    def add_bess(
            self
    ):
        """Add storages elements."""
        for battery in self.bess_attrs:
            self.set_battery(**battery)

    def remove_batteries(
            self
    ):
        """Turn off BESS elements.

        .. warning::
            Disable element by full name.

        """
        dssStorage = self.dss.ActiveCircuit.Storages
        for storage_id in self.storages_id:
            dssStorage.Name = storage_id   # Activate
            dssobj = self.dss.ActiveCircuit.ActiveElement
            try:
                if not dssobj.Enabled:
                    message: str = (
                        f"The element '{storage_id}' it is "
                        "not currently in the circuit"
                    )
                    raise ValueError(message)
            except ValueError as e:
                print(f"AlreadyDisabled: {e}.")
            else:
                full_name = dssobj.Name
                self.dss.ActiveCircuit.Disable(full_name)

    def deploy_pv_monitors(
            self,
            polar: bool = True
    ):
        """Embed voltage-current monitor to PV systems."""
        iPVsys = self.dss.ActiveCircuit.PVSystems
        pv_systems: list[str] = iPVsys.AllNames
        pv_systems = [
            None if (p.lower() in {"none", "nan", "null"}) else p
            for p in pv_systems if p
        ]
        pv_systems = list(filter(None, pv_systems))

        for id in pv_systems or []:
            monitor_id = self.add_monitor(
                full_name_element=f"PVSystem.{id}",
                monitor_id=f"pv_monitor_{id}_vi",
                terminal=1,
                mode=enums.MonitorModes.VI,
                polar=polar
            )
            self.volt_curr_der_monitors.append(monitor_id)

    def set_pv_loadshape(
            self,
            shape_id: str,
            npts: int = 96,
            minterval: int = 15,
            irradiance: np.ndarray | None = None,
            window_hours: tuple[float, float] | None = None
    ):
        """Create a LoadShape object to represent irradiance / daily PV shape.

        If ``irradiance`` (array-like) provided, it is
        used directly (length must match npts). Else if ``window_hours``
        provided, create a triangular/simple window across those hours.
        Otherwise create a flat shape of ones.

        """
        dssLoadShape = self.dss.ActiveCircuit.LoadShapes
        dssLoadShape.New(shape_id)
        dssLoadShape.Name = shape_id
        dssLoadShape.Npts = npts
        dssLoadShape.MinInterval = minterval
        dssLoadShape.UseActual = False

        if irradiance is not None:
            arr = np.asarray(irradiance, dtype=float)
            try:
                if arr.size != npts:
                    message: str = (
                        f"Irradiance length ({arr.size}) != "
                        f"npts ({npts})"
                    )
                    raise ValueError(message)
            except ValueError as e:
                print(f"InvalidLoadshape: {e}")
                return
            else:
                pmult = arr
        elif window_hours:
            # Create a smooth window (raised cosine) in the given hours
            pmult = np.zeros(npts)
            start = int(npts * window_hours[0] / 24.0)
            stop = int(npts * window_hours[1] / 24.0)
            if stop <= start:
                stop = start + 1
            width = stop - start
            x = np.linspace(-np.pi/2, np.pi/2, width)
            window = np.cos(x) ** 2  # smooth bell-like
            pmult[start:stop] = window
        else:
            pmult = np.ones(npts)

        dssLoadShape.Pmult = pmult

    def set_pv_sys(
                self,
                pv_id: str,
                bus_id: str,
                phases: int = 3,
                kV: float = 0.4,
                kVA: float | None = None,
                pf: float = 1.0,
                conn: str = "wye",
                model: int = 1,
                daily_shape_id: str | None = None,
                irradiance: np.ndarray[float] = None,
                window_hours: tuple[float, float] = None,
                npts: int = 96,
                minterval: int = 15,
                pmpp: float = None,
                enabled: bool = True,
                **extra
    ):
        """Create a PVSystem and optionally a daily irradiance/loadshape.

        Parameters
        ----------

        pv_id : str
            identifier (will create PVSystem.<pv_id>).
        bus_id : str
            bus to connect (string).
        phases , int
            kV, kW: electrical sizing.
        kVA : float
            inverter rating; if None, set to kW (assumes pf=1).
        pf : float
            power factor for generator mode (if using Pgen type behavior).
        daily_shape_id : str
            name of existing LoadShape; if not given and irradiance
            or window provided, a shape will be created.
        irradiance : np.ndarray[float]
            optional array for shape creation.
        window_hours : tuple[float, float]
            optional simple daily window (e.g. (6, 18)).
        pmpp : float
            peak-power point parameter (optional).
        extra : **kwargs
            forwarded to DSS command as additional properties.

        """
        if (daily_shape_id is None) and (
            irradiance is not None or window_hours is not None
        ):
            daily_shape_id = f"{pv_id}_irradiance"
            self.set_pv_loadshape(
                daily_shape_id,
                npts=npts,
                minterval=minterval,
                irradiance=irradiance,
                window_hours=window_hours
            )

        # Build DSS command; include commonly used parameters
        cmd_parts = [
            f"New PVSystem.{pv_id}",
            f"phases={phases}",
            f"bus1={bus_id}",
            f"kV={kV}",
            f"kVA={kVA}",
            f"%cutin=0",      # Defaults - can be overridden via extra
            f"conn={conn}",
            f"model={model}"
        ]
        if daily_shape_id:
            cmd_parts.append(f"daily={daily_shape_id}")
        if pmpp is not None:
            cmd_parts.append(f"pmpp={pmpp}")
        if pf is not None:
            cmd_parts.append(f"pf={pf}")

        # Extra keyword args forwarded as key=value
        for k, v in extra.items():
            cmd_parts.append(f"{k}={v}")

        cmd = " ".join(cmd_parts)
        self.dss.Text.Command = cmd

        # Validate creation and optionally enable/state handling
        try:
            # Select PVSystem by name and check active element
            self.dss.ActiveCircuit.SetActiveElement(f"PVSystem.{pv_id}")
            active_name = self.dss.ActiveCircuit.ActiveElement.Name
            if active_name.lower() != f"pvsystem.{pv_id}".lower():
                message: str = (
                    f"PVSystem {pv_id} not found after creation "
                    f"(active element {active_name})"
                )
                raise ValueError(message)
        except Exception as e:
            logg: str = (
                "ElementNotFound or creation error for PVSystem "
                f"{pv_id}: {e}"
            )
            print(logg)
            return
        else:
            if not enabled:
                # Disable the element if user requested
                self.dss.ActiveCircuit.Disable(active_name)

            self.pvsystems_id.append(f"PVSystem.{pv_id}")

    def add_pv_systems(
            self
    ):
        """Spread Photovoltaic systems throughout circuit."""
        for pv_sys in self.pvsys_attrs:
            self.set_pv_sys(**pv_sys)

    def remove_pv_systems(
            self
    ):
        """Turn off PVSystems.

        .. warning::
            Disable element by full name.

        """
        dssPV = self.dss.ActiveCircuit.PVSystems
        for pv_id in self.pvsystems_id:
            dssPV.Name = pv_id   # Activate
            dssobj = self.dss.ActiveCircuit.ActiveElement
            try:
                if not dssobj.Enabled:
                    message: str = (
                        f"The element '{pv_id}' it is "
                        "not currently in the circuit"
                    )
                    raise ValueError(message)
            except ValueError as e:
                print(f"AlreadyDisabled: {e}.")
            else:
                full_name = dssobj.Name
                self.dss.ActiveCircuit.Disable(full_name)

    def calculate_losses(
            self
    ) -> np.ndarray:
        """Return and set global network losses."""
        delta_matrix = self.local_mismatch()
        external_gen = self.external_network_power()
        delta_matrix += external_gen
        self.ckt_losses = delta_matrix
        return delta_matrix

    def fault_network(
            self
    ):
        """Run fault study of short circuit all across the circuit.

        Filter out fault types regarding the number of nodes
        of bus. Finally add up DER short current contribution.

        Classify MV buses and LV ones.

        .. Note::
            Above voltage magnitude of 30kV LL it is considered
            medium voltage.

        .. warning::
            Both ``Vsourcebus`` and renamed third
            floating winding ``hvmv_3`` are skiped.

        """
        fault_buses = {}
        for bus_id in self.dss.ActiveCircuit.AllBusNames:
            dssBus = self.dss.ActiveCircuit.ActiveBus(bus_id)
            skip_bus = (
                dssBus.Name not in self.ckt_graph.vertices,
                bus_id.lower() in {"sourcebus", "hvmv_3"}
            )
            if any(skip_bus):
                continue
            nodes = dssBus.Nodes
            voltages = dssBus.VMagAngle  # mag VLN [V], phase [Deg]
            if all(voltages[::2] >= 30.0e3/np.sqrt(3)):
                self.mv_buses_id.append(bus_id)
            else:
                self.lv_buses_id.append(bus_id)

            terminals = [f".{n}" for n in nodes if n]
            if len(nodes) == 1:
                fault_buses[bus_id] = (['LG'], terminals)
            else:
                fault_buses[bus_id] = (
                    ['LG', 'LLG_a', 'LLG_b', 'LL'], terminals
                )
                if len(nodes) >= 3:
                    fault_buses[bus_id][0].append('LLLG')

        self.run_fault_study(**fault_buses)
        return self.ckt_faults


if __name__ == "__main__":
    pass
