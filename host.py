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
        ...

    @abstractmethod
    def fault_network(self):
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
        self.allocate_layers_color()  # In place

    def read_gis(
            self,
            path: str,
            epsg: int = 5367,
            inplace: bool = False
    ) -> gpd.GeoDataFrame:
        """Set the leaf up.

        Ensure CRS is EPSG:4326 for folium compatibility.
        Returns a copy to EPSG:5367 due to
        utility-scale electrical modeling in Costa Rica
        if inplaced.

        """
        try:
            gdf = gpd.read_file(path)
        except TypeError as e:
            print(f"ErrorReadingFile: {path}. {e.args}")
        else:
            if gdf.crs.to_epsg() != epsg:
                return (
                    gdf.to_crs(epsg=epsg) if inplace else gdf.to_crs(epsg=4326)
                )

    def get_layers(
            self,
    ):
        """Fill layers up.

        Shapefile name may be: ``<xxxx>_<name>``.
    
        """
        shapefiles = glob.glob(self.gis_path)
        for shp in shapefiles:
            name = shp.split("/")[-1].split(".")[0].split("_")[-1]
            self.layers[name] = [self.read_gis(shp)]

    def allocate_layers_color(
            self,
            seed: int = 7859
    ) -> dict[str, list[str, str]]:
        """Asign eye-cathing color to each layer.

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
        """Map of the circuit."""
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
class Graph(GISCircuit):
    """Topology analysis."""

    pass


@dataclass()
class Network(ABC):
    """Abstract Base Class of Circuits (Manufacturer)."""

    ckt_path: str = "./CKT/CKT_Daily.dss"
    solve_mode: int = enums.SolveModes.Daily
    control_mode: int = enums.ControlModes.Time
    algorithm: int = enums.SolutionAlgorithms.NewtonSolve
    number: int = 96
    stepsize_min: int = 15
    to_solve: bool = True
    feeders_power: np.ndarray[float] | None = None
    der_data = {"Storage": [], "PVSystem": []}
    der_names: list = field(default_factory=list)
    der_dummy_names: list = field(default_factory=list)
    power_monitors_id:  list[str] = field(default_factory=list)
    losses_monitors_id:  list[str] = field(default_factory=list)
    head_meters_id:  list[str] = field(default_factory=list)
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

    def set_monitor(
            self,
            full_name_element: str = "transformer.substation",
            monitor_id: str = "substation_monitor_1",
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power
    ) -> str:
        """Instantiate and set a single monitor."""
        dssMonitors = self.dss.ActiveCircuit.Monitors
        monitors_id = dssMonitors.AllNames
        if monitor_id in monitors_id:
            return monitor_id

        self.dss.Text.Command = f"new monitor.{monitor_id} ppolar=no"
        dssMonitors.Name = monitor_id
        dssMonitors.Element = full_name_element
        dssMonitors.Terminal = terminal
        dssMonitors.Mode = mode
        return monitor_id

    def set_meter(
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

    def add_head_monitors(
            self,
            source_bus_id: str = "sourcebus",
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power
    ):
        """Deploy monitors to each PDE connected to sourcebus.

        To keep an eye on external network.

        """
        self.dss.ActiveCircuit.ActiveBus(source_bus_id)
        # Full name branches
        feeder_branches = self.dss.ActiveCircuit.ActiveBus.AllPDEatBus
        # Branches at source bus
        for branch in feeder_branches:
            if branch:
                _ = self.dss.ActiveCircuit.SetActiveElement(branch)
                element_id = self.dss.ActiveClass.Name
                monitor_id = self.set_monitor(
                    branch, f"{element_id}_monitor_{mode}", terminal, mode
                )
                self.power_monitors_id.append(monitor_id)

    def add_head_meters(
            self,
            source_bus_id: str = "sourcebus",
            terminal: int = 1,
    ):
        """Embed EnergyMeter right at feeders head.

        To assess Topology analysis and collect global Registers.

        """
        self.dss.ActiveCircuit.ActiveBus(source_bus_id)
        feeder_branches = self.dss.ActiveCircuit.ActiveBus.AllPDEatBus
        for branch in feeder_branches:
            if branch:
                _ = self.dss.ActiveCircuit.SetActiveElement(branch)
                element_id = self.dss.ActiveClass.Name
                meter_id = self.set_meter(
                    branch, f"{element_id}_meter", terminal
                )
                self.head_meters_id.append(meter_id)

    def get_monitor_data(
            self,
            monitor_id: str = "feeder_pq",
            reset: bool = True
    ):
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
            terminal: int = 1,
            mode: int = enums.MonitorModes.Power
    ):
        """Connect measurement infrastructure to PCE.

        Power Conversion Elements (PCE) regarding
        the local network in order to measure losses.

        .. warning::
            Ensure to call this method in the proper
            monitoring kind of mode.

        .. Note::
            DER devices full names are internally retained.

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
                monitor_id = self.set_monitor(
                    full_name,
                    f"{element_id}_monitor_{mode}",
                    terminal,
                    mode
                )
                self.losses_monitors_id.append(monitor_id)
                i = self.dss.ActiveCircuit.NextPCElement()

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
                for name in self.power_monitors_id:
                    data = self.get_monitor_data(name)
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
            self.feeders_power = injected_power
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
            absorves reactive [kVAr].

        Returns
        -------
        delta_matrix : np.ndarray[float]
            Switched sign so positive remaining either
            actitive or reactive it is seen as a surplus.

        .. Note::
            No external network contribution.

        """
        delta_matrix = np.zeros(
            (self.dss.ActiveCircuit.Solution.Number, 2)
        )
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                for name in self.losses_monitors_id:
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
class BaseCicuit(Network, Circuit):
    """Current circuit."""

    def __post_init__(self):
        super().__post_init__()
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                self.add_head_meters()
                self.add_head_monitors()          # Power monitors
                self.deploy_pce_monitors(mode=9)  # Losses monitors
                self.dss.ActiveCircuit.Solution.Solve()
            else:
                raise RuntimeError("Circuit must be initialized")
        except RuntimeError as e:
            print(f"NonSolvedCkt: {e}.")

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
            if "sourcebus" in bus_id.lower():
                continue
            if "hvmv_3" in bus_id.lower():
                continue
            dssBus = self.dss.ActiveCircuit.ActiveBus(bus_id)
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

    def __post_init__(self):
        """Add DER and then monitors."""
        super().__post_init__()
        try:
            if self.dss.ActiveCircuit.Solution.Converged:
                # -- DER
                self.add_bess()
                # -- Measuring
                self.add_head_meters()
                self.add_head_monitors()          # Power monitors
                self.deploy_pce_monitors(mode=9)  # Losses monitors
                self.dss.ActiveCircuit.Solution.Solve()
            else:
                raise RuntimeError("Circuit must be initialized")
        except RuntimeError as e:
            print(f"NonSolvedCkt: {e}.")

    def set_bess_dispatch_curve(
            self,
            dispatch_curve_id: str = "dispatch_shape",
            npts: int = 96,
            minterval: int = 15,
            hours_soc: tuple[tuple, tuple] = ((1, 5), (18, 22))
    ):
        """Define dynamically LoadShape.

        Generic daily dispatch shape curve of storage device.

        Parameters
        ----------
        hours_soc : tuple[tuple, tuple]
            Time boundaries for the State of Charge (SoC) of
            storage device where the outer tuple defines
            if discharge or charge and the inner
            set the hours from and to as **24-hr** fashion.

        """
        dssLoadShape = self.dss.ActiveCircuit.LoadShapes
        dssLoadShape.New(dispatch_curve_id)
        dssLoadShape.Name = dispatch_curve_id
        dssLoadShape.Npts = npts
        dssLoadShape.MinInterval = minterval
        dssLoadShape.UseActual = False
        # Charge (negative values)
        i = int(npts*hours_soc[0][0] / 24.0)
        j = int(npts*hours_soc[0][1] / 24.0)
        # Discharge (positive values)
        m = int(npts*hours_soc[1][0] / 24.0)
        n = int(npts*hours_soc[1][1] / 24.0)
        daily_dispatch = np.zeros(npts)
        daily_dispatch[i:j] = -1.0
        daily_dispatch[m:n] = 1.0
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
            triggers: tuple[float] = (0.95, 0.20)
    ):
        """Integrate BESS to the circuit."""
        self.set_bess_dispatch_curve(daily_id)
        self.dss.Text.Command = (
            f"New Storage.{storage_id} phases={phases} "
            f"bus1={bus_id} kV={kV} "
            f"kWrated={kW} kWhrated={kWh} %stored={per_stored} "
            f"%reserve={per_reserve} "
            f"%effcharge={per_efficiencies[0]} "
            f"%effdischarge={per_efficiencies[1]} "
            f"DischargeTrigger={triggers[0]} ChargeTrigger={triggers[1]} "
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

    def set_pv_sys(
            self
    ):
        """Creat PVSystem element within current context."""
        return

    def add_pv_systems(
            self
    ):
        """Spread Photovoltaic systems throughout circuit."""
        for pv_sys in self.pvsys_attrs:
            self.set_pv_sys(**pv_sys)

    def remove_batteries(
            self
    ):
        """Turn off BESS elements.

        .. warning::
            Disable element thorugh full name.

        """
        dssStorage = self.dss.ActiveCircuit.Storages
        for storage_id in self.storages_id:
            dssStorage.Name = storage_id   # Activate
            full_name = self.dss.ActiveCircuit.ActiveElement.Name
            self.dss.ActiveCircuit.Disable(full_name)

    def remove_pv_systems(
            self
    ):
        """Turn off PVSystems

        .. warning::
            Disable element thorugh full name.

        """

        for pv_id in self.pvsystems_id:
            self.dss.ActiveCircuit.Disable(pv_id)

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
            if "sourcebus" in bus_id.lower():
                continue
            if "hvmv_3" in bus_id.lower():
                continue
            dssBus = self.dss.ActiveCircuit.ActiveBus(bus_id)
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
