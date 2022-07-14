// FORLORN - Comparing Offline Methods and RL for RAN Parameter Optimization
// Copyright (c) 2022 Telenor ASA
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 2 as
// published by the Free Software Foundation.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
//
// -----
//
// This code accompanies the following paper: Vegard Edvardsen, Gard Spreemann,
// and Jeriek Van den Abeele. 2022. FORLORN: A Framework for Comparing Offline
// Methods and Reinforcement Learning for Optimization of RAN Parameters.
// Submitted to the 25th ACM International Conference on Modeling, Analysis and
// Simulation of Wireless and Mobile Systems (MSWiM '22).
//
// -----
//
// simulator.cc - Network simulation scenario
//
// This program uses the ns-3 library to implement the network simulation
// scenario we want to optimize. The scenario contains a number of LTE base
// stations (cells) and users (UEs, identified by their IMSI). The goal is to
// find the base station configuration that offers the best throughput for the
// users. While this simulation is implemented in C++, the rest of the pipeline
// (Reinforcement Learning agent, grid search and Optuna optimized baselines,
// visualization code) is implemented in Python. The module simulator.py is the
// interface between these two domains, using human-readable text over standard
// input/output pipes as the medium for inter-process communication

#include "ns3/core-module.h"
#include "ns3/lte-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/internet-apps-module.h"
#include "ns3/applications-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-helper.h"
#include <cmath>
#include <random>
#include <tuple>

#include "mobility.h"

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("NetworkScenario");

//------------------------------------------------------------------------------
// Base class for scenario (subclassed by real and mock network scenario)
//------------------------------------------------------------------------------

class BaseNetworkScenario
{
    public:
        void initialize(
            int warmup_duration, int interaction_interval, double enb_inter_distance,
            int left_enb_power, int right_enb_power, int top_enb_power,
            int ue_count_min, int ue_count_max, int total_ue_count, int seed_pause,
            const std::vector<std::uint64_t> & seeds, double ue_speed,
            int min_power, int max_power);
        void run();

        virtual void generate_map(std::string map_file) = 0;
        virtual void enable_traces() = 0;

    protected:
        int warmup_duration;
        int interaction_interval;
        double enb_inter_distance;
        double map_extent;

        struct enb_configuration { double x, y, z, direction; int tx_power; bool tx_legal; };
        std::vector<enb_configuration> enb_conf;

        std::vector<std::vector<ns3::Vector>> ue_waypoints;
        std::vector<std::uint64_t> seeds;
        int seed_pause;
        double ue_speed;
        int min_power;
        int max_power;
        int seed_number;
        int total_ue_count;
        ns3::Time last_ue_arrival;
        void ue_depart_callback();
        void ue_arrive_callback();

        NodeContainer enb_nodes;
        NodeContainer ue_nodes;

        void generate_enb_conf(int left_enb_power, int right_enb_power, int top_enb_power);
        void define_ue_clusters(const std::vector<std::uint64_t> & seeds, int ue_count_min, int ue_count_max);
        void create_enb_nodes();
        void create_ue_nodes();

        virtual void create_lte_network() = 0;
        virtual void apply_network_conf() = 0;
        virtual void create_remote_server() = 0;
        virtual void create_ue_applications() = 0;
        virtual void setup_callbacks() = 0;
        virtual int get_ue_rx_bytes(int i) = 0;

        void dump_initial_state();
        void periodically_interact_with_agent();

        int timestep() { return Simulator::Now().GetMilliSeconds() - this->warmup_duration;  }
};

void BaseNetworkScenario::initialize(
        int warmup_duration, int interaction_interval, double enb_inter_distance,
        int left_enb_power, int right_enb_power, int top_enb_power,
        int ue_count_min, int ue_count_max, int total_ue_count, int seed_pause,
        const std::vector<std::uint64_t> & seeds, double ue_speed,
        int min_power, int max_power)
{
    this->seed_number = 0;
    this->total_ue_count = total_ue_count;
    this->seed_pause = seed_pause;
    this->ue_speed = ue_speed;
    this->min_power = min_power;
    this->max_power = max_power;

    this->warmup_duration = warmup_duration;
    this->interaction_interval = interaction_interval;
    this->enb_inter_distance = enb_inter_distance;

    this->generate_enb_conf(left_enb_power, right_enb_power, top_enb_power);
    this->define_ue_clusters(seeds, ue_count_min, ue_count_max);
    this->create_enb_nodes();
    this->create_ue_nodes();

    this->create_lte_network();
    this->apply_network_conf();
    this->create_remote_server();
    this->create_ue_applications();
    this->setup_callbacks();
}

void BaseNetworkScenario::run()
{
    this->dump_initial_state();
    this->periodically_interact_with_agent();
    Simulator::Run();
    Simulator::Destroy();
}

void BaseNetworkScenario::generate_enb_conf(
        int left_enb_power, int right_enb_power, int top_enb_power)
{
    // The configured eNodeB distance (this->enb_inter_distance) corresponds to
    // the sides of an equilateral triangle. We wish to place the eNodeBs in a
    // fixed radius around (0, 0), so first we need to calculate the circumradius
    // of the equilateral triangle based on the side length
    double r = this->enb_inter_distance / 2 / cos(30 * M_PI / 180);

    this->enb_conf.resize(3);
    this->enb_conf[0] = { r * cos(210 * M_PI / 180), r * sin(210 * M_PI / 180), 30.0, 210 - 180,
      left_enb_power, left_enb_power >= this->min_power && left_enb_power <= this->max_power };
    this->enb_conf[1] = { r * cos(-30 * M_PI / 180), r * sin(-30 * M_PI / 180), 30.0, -30 + 180,
      right_enb_power, right_enb_power >= this->min_power && right_enb_power <= this->max_power };
    this->enb_conf[2] = { r * cos( 90 * M_PI / 180), r * sin( 90 * M_PI / 180), 30.0,  90 + 180,
      top_enb_power, top_enb_power >= this->min_power && top_enb_power <= this->max_power };

    this->map_extent = 0;
    for (uint32_t i = 0; i < this->enb_conf.size(); i++) {
        this->map_extent = std::fmax(this->map_extent, std::abs(this->enb_conf[i].x));
        this->map_extent = std::fmax(this->map_extent, std::abs(this->enb_conf[i].y));
    }
}

void BaseNetworkScenario::define_ue_clusters(const std::vector<std::uint64_t> & seeds, int ue_count_min, int ue_count_max)
{
  this->seeds = seeds;

  for (auto seed : seeds)
  {
    std::mt19937_64 rng;
    rng.seed(seed);
    int ues_remaining = total_ue_count;

    this->ue_waypoints.push_back(std::vector<ns3::Vector>());

    while (ues_remaining > 0)
    {
      std::uniform_int_distribution<> size_distr(ue_count_min, ue_count_max);
      int ue_count = std::min(ues_remaining, size_distr(rng));

      std::uniform_real_distribution<> radius_distr(enb_inter_distance/50.0, enb_inter_distance/2.0);
      double radius = radius_distr(rng);

      double box_width = this->enb_inter_distance/2.0;
      std::uniform_real_distribution<> box_distr(-box_width + radius, box_width - radius);
      double c_x = box_distr(rng);
      double c_y = box_distr(rng);

      std::uniform_real_distribution<> x_distr(c_x - radius, c_x + radius);
      std::uniform_real_distribution<> y_distr(c_y - radius, c_y + radius);

      for (int i = 0; i < ue_count; ++i)
      {
        double x = x_distr(rng);
        double y = y_distr(rng);
        while (std::pow(x - c_x, 2.0) + std::pow(y - c_y, 2.0) > std::pow(radius, 2.0))
        {
          x = x_distr(rng);
          y = y_distr(rng);
        }
        this->ue_waypoints.back().push_back(ns3::Vector(x, y, 0.0));
      }
      
      ues_remaining -= ue_count;
    }
  }
}

void BaseNetworkScenario::create_enb_nodes()
{
    this->enb_nodes.Create(this->enb_conf.size());
    MobilityHelper mobility_helper;
    mobility_helper.Install(this->enb_nodes);

    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        Ptr<Node> enb_node = this->enb_nodes.Get(i);
        Ptr<MobilityModel> mobility = enb_node->GetObject<MobilityModel>();
        mobility->SetPosition(Vector(this->enb_conf[i].x, this->enb_conf[i].y, this->enb_conf[i].z));
    }
}

void BaseNetworkScenario::create_ue_nodes()
{
  NodeContainer new_ue_nodes;
  new_ue_nodes.Create(total_ue_count);
  this->ue_nodes.Add(new_ue_nodes);

  MobilityHelper mobility_helper;
  mobility_helper.SetMobilityModel("forlorn::DirectMobilityModel");
  mobility_helper.Install(this->ue_nodes);

  NS_ASSERT(!ue_waypoints.empty());
  const std::vector<ns3::Vector> & initial_positions = ue_waypoints[0];
  for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
    Ptr<forlorn::DirectMobilityModel> mobility = this->ue_nodes.Get(i)->GetObject<forlorn::DirectMobilityModel>();
    mobility->SetAttribute("Speed", DoubleValue(ue_speed));
    mobility->SetAttribute("From", VectorValue(initial_positions[i]));
    mobility->SetAttribute("To", VectorValue(initial_positions[i]));
  }
  Simulator::Schedule(MilliSeconds(this->warmup_duration + this->seed_pause), &BaseNetworkScenario::ue_depart_callback, this);
}

void BaseNetworkScenario::ue_depart_callback() {
  if (seed_number + 1 < ue_waypoints.size()) {
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
      Ptr<forlorn::DirectMobilityModel> mobility = this->ue_nodes.Get(i)->GetObject<forlorn::DirectMobilityModel>();
      mobility->SetAttribute("To", VectorValue(ue_waypoints[seed_number + 1][i]));
      if (mobility->get_arrival_time() > last_ue_arrival) {
        last_ue_arrival = mobility->get_arrival_time();
      }
    }
    ns3::Time now = Simulator::Now();
    if (last_ue_arrival > now) {
      Simulator::Schedule(last_ue_arrival - now, &BaseNetworkScenario::ue_arrive_callback, this);
    }
    std::cout << this->timestep() << " ms: Seed " << this->seeds[seed_number] << "-" << this->seeds[seed_number + 1] << std::endl;
    ++seed_number;
  }
}

void BaseNetworkScenario::ue_arrive_callback() {
    std::cout << this->timestep() << " ms: Seed " << this->seeds[seed_number] << std::endl;
    Simulator::Schedule(MilliSeconds(this->seed_pause), &BaseNetworkScenario::ue_depart_callback, this);
}

void BaseNetworkScenario::dump_initial_state()
{
    // Upon start of simulation, dump position and orientation of each eNodeB
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        Ptr<Node> node = this->enb_nodes.Get(i);
        Vector position = node->GetObject<MobilityModel>()->GetPosition();
        double direction = this->enb_conf[i].direction;
        std::cout << this->timestep() << " ms: Cell state: "
            << "Cell " << (i + 1)
            << " at " << position.x << " " << position.y
            << " direction " << direction << std::endl;
    }

    // Also dump the random number generator seed used to generate UE clusters
    std::cout << this->timestep() << " ms: Seed " << this->seeds[0] << std::endl;
}

void BaseNetworkScenario::periodically_interact_with_agent()
{
    // Dump relevant simulation state for each UE to stdout. Currently we are
    // interested in 2D position and IPv4 bytes received since last time
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        Ptr<Node> node = this->ue_nodes.Get(i);
        Vector position = node->GetObject<MobilityModel>()->GetPosition();
        std::cout << this->timestep() << " ms: UE state: "
            << "IMSI " << (i + 1)
            << " at " << position.x << " " << position.y
            << " with " << this->get_ue_rx_bytes(i) << " received bytes" << std::endl;
    }

    // Dump the current cell parameter configuration to stdout
    std::cout << this->timestep() << " ms: Configuration: Cell";
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        std::cout << " tx" << (i + 1) << " " << this->enb_conf[i].tx_power;
    }
    std::cout << std::endl;

    // Dump the current tx power legality status to stdout
    std::cout << this->timestep() << " ms: Configuration legality: Cell";
        for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        std::cout << " tx" << (i + 1) << " " << this->enb_conf[i].tx_legal;
    }
    std::cout << std::endl;

    // Only ask for new cell parameters from the agent if the warmup phase is
    // over (in which case this->timestep() will return a non-negative number)
    if (this->timestep() >= 0) {
        // Read in the new transmission power levels for all three cells
        std::cout << this->timestep() << " ms: Agent action?" << std::endl;
        int power = 0;
        for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
            if (!(std::cin >> power)) {
                throw std::invalid_argument("Invalid action input");
            }
            if (power < this->min_power || power > this->max_power) {
                power = std::min(std::max(power, this->min_power), this->max_power);
                this->enb_conf[i].tx_legal = false;
            }
            else {
                this->enb_conf[i].tx_legal = true;
            }
            this->enb_conf[i].tx_power = power;
        }
        // Call the subclass-specific configuration update method
        this->apply_network_conf();
    }

    // Reschedule again after this->interaction_interval (default 100 ms)
    Simulator::Schedule(MilliSeconds(this->interaction_interval),
        &BaseNetworkScenario::periodically_interact_with_agent, this);
}

//------------------------------------------------------------------------------
// Subclass for the real network scenario
//------------------------------------------------------------------------------

class RealNetworkScenario : public BaseNetworkScenario
{
    public:
        void generate_map(std::string map_file) override;
        void enable_traces() override;

    protected:
        Ptr<LteHelper> lte_helper;
        Ptr<EpcHelper> epc_helper;
        NodeContainer server_nodes;
        std::vector<PacketSizeMinMaxAvgTotalCalculator*> ue_packet_calcs;

        void create_lte_network() override;
        void apply_network_conf() override;
        void create_remote_server() override;
        void create_ue_applications() override;
        void setup_callbacks() override;
        int get_ue_rx_bytes(int i) override;

        static void callback_ipv4_packet_received(
            PacketSizeMinMaxAvgTotalCalculator* packet_calc,
            Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t iface);
        void callback_ue_spotted_at_enb(
            std::string context, const uint64_t imsi,
            const uint16_t cell_id, const uint16_t rnti);
        void callback_measurement_report_received(
            const uint64_t imsi, const uint16_t cell_id,
            const uint16_t rnti, const LteRrcSap::MeasurementReport report);
};

void RealNetworkScenario::generate_map(std::string map_file)
{
    Ptr<RadioEnvironmentMapHelper> rem_helper = CreateObject<RadioEnvironmentMapHelper>();
    rem_helper->SetAttribute("ChannelPath", StringValue("/ChannelList/2"));
    rem_helper->SetAttribute("XMin", DoubleValue(-this->map_extent));
    rem_helper->SetAttribute("XMax", DoubleValue(this->map_extent));
    rem_helper->SetAttribute("YMin", DoubleValue(-this->map_extent));
    rem_helper->SetAttribute("YMax", DoubleValue(this->map_extent));
    rem_helper->SetAttribute("OutputFile", StringValue(map_file));
    rem_helper->Install();

    // Start the simulator to trigger the rendering
    Simulator::Run();
    Simulator::Destroy();
}

void RealNetworkScenario::enable_traces()
{
    this->lte_helper->EnableTraces();
}

void RealNetworkScenario::create_lte_network()
{
    // Create LTE and EPC helpers. Network to be set up as a bunch of LTE base
    // stations (eNodeB), attached to an EPC (network core) implementation and
    // UEs (mobile handsets)
    this->epc_helper = CreateObject<PointToPointEpcHelper>();
    this->lte_helper = CreateObject<LteHelper>();
    this->lte_helper->SetEpcHelper(this->epc_helper);

    // Set up a directional antenna, to allow 3-sector base stations
    this->lte_helper->SetEnbAntennaModelType("ns3::ParabolicAntennaModel");
    this->lte_helper->SetEnbAntennaModelAttribute("Beamwidth", DoubleValue(70.0));

    // Activate handovers using a default RSRQ-based algorithm
    this->lte_helper->SetHandoverAlgorithmType("ns3::A2A4RsrqHandoverAlgorithm");

    // Select "hard" frequency reuse (FR), which fully partitions the spectrum
    // into three equal parts and distributes those among the base stations
    this->lte_helper->SetFfrAlgorithmType("ns3::LteFrHardAlgorithm");

    // Specify that the RLC layer of the LTE stack should use Acknowledged Mode
    // (AM) as the default mode for all data bearers. This as opposed to the
    // ns-3 default which is Unacknowledged Mode (UM), see lte-enb-rrc.cc:1699
    // and lte-helper.cc:618. This is important because TCP traffic between a
    // UE and a remote host is very sensitive to packet loss. Packets lost
    // between the UE and eNodeB will be treated by TCP as a signal that the
    // network is congested -- but it might simply be that the radio conditions
    // are bad! RLC AM mode ensures reliable delivery across the radio link,
    // relieving TCP of that responsibility and not triggering any congestion
    // control algorithms in TCP. This greatly improves TCP performance
    Config::SetDefault("ns3::LteEnbRrc::EpsBearerToRlcMapping",
        EnumValue(LteEnbRrc::RLC_AM_ALWAYS));

    // Bump the maximum possible number of UEs connected per eNodeB
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(80));

    // Loop through the eNodeB nodes and set up the base stations
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        Ptr<Node> node = this->enb_nodes.Get(i);
        this->lte_helper->SetEnbAntennaModelAttribute(
            "Orientation", DoubleValue(this->enb_conf[i].direction));
        this->lte_helper->SetFfrAlgorithmAttribute(
            "FrCellTypeId", UintegerValue((i % 3) + 1));
        this->lte_helper->InstallEnbDevice(node);
    }

    // Add an X2 interface between the eNodeBs, to enable handovers
    this->lte_helper->AddX2Interface(this->enb_nodes);
}

void RealNetworkScenario::apply_network_conf()
{
    // Set base station transmission powers according to chosen values
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        std::ostringstream oss;
        oss << "/NodeList/" << this->enb_nodes.Get(i)->GetId();
        oss << "/DeviceList/*/ComponentCarrierMap/*/LteEnbPhy/TxPower";
        Config::Set(oss.str(), DoubleValue(0.1 * this->enb_conf[i].tx_power));
    }
}

void RealNetworkScenario::create_remote_server()
{
    // Create the server that will send downlink traffic to UEs
    this->server_nodes.Create(1);
    InternetStackHelper ip_stack_helper;
    ip_stack_helper.Install(this->server_nodes);

    // Connect the server to the PDN gateway (PGW) in the EPC
    PointToPointHelper p2p_helper;
    p2p_helper.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gbps")));
    p2p_helper.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer server_devices = p2p_helper.Install(
        this->server_nodes.Get(0), this->epc_helper->GetPgwNode());

    // Set up IP interfaces on the link between PGW and the server
    Ipv4AddressHelper ipv4_helper("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer server_ifaces = ipv4_helper.Assign(server_devices);

    // Add an IP route on the server toward the PGW interface, for the UE subnet (7.0.0.0/8)
    Ipv4StaticRoutingHelper routing_helper;
    Ptr<Ipv4StaticRouting> server_routing = routing_helper.GetStaticRouting(
        this->server_nodes.Get(0)->GetObject<Ipv4>());
    int server_iface_toward_pgw = server_ifaces.Get(0).second;
    server_routing->AddNetworkRouteTo("7.0.0.0", "255.0.0.0", server_iface_toward_pgw);
}

void RealNetworkScenario::create_ue_applications()
{
    // Add a default IP stack to the UEs. The EPC helper will later
    // assign addresses to UEs in the 7.0.0.0/8 subnet by default
    InternetStackHelper ip_stack_helper;
    ip_stack_helper.Install(this->ue_nodes);

    // Create UE net devices and assign IP addresses in EPC
    NetDeviceContainer ue_devices = this->lte_helper->InstallUeDevice(this->ue_nodes);
    Ipv4InterfaceContainer ue_ifaces = this->epc_helper->AssignUeIpv4Address(ue_devices);

    // Attach the UEs to the LTE network
    this->lte_helper->Attach(ue_devices);

    // Set default IP route for all UEs
    Ipv4StaticRoutingHelper routing_helper;
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        routing_helper.GetStaticRouting(this->ue_nodes.Get(i)->GetObject<Ipv4>())
            ->SetDefaultRoute(this->epc_helper->GetUeDefaultGatewayAddress(), 1);
    }

    // Set up CBR (constant bitrate) traffic generators from the server to each UE
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        const char *socket_factory_type = "ns3::TcpSocketFactory";
        InetSocketAddress cbr_dest(ue_ifaces.GetAddress(i), 10000);
        OnOffHelper cbr_helper(socket_factory_type, cbr_dest);
        cbr_helper.SetConstantRate(DataRate("20Mbps"));
        ApplicationContainer cbr_apps = cbr_helper.Install(this->server_nodes.Get(0));
        cbr_apps.Start(Seconds(1));

        // Set up a TCP/UDP sink on the receiving side (UE)
        PacketSinkHelper packet_sink_helper(socket_factory_type, cbr_dest);
        ApplicationContainer sink_apps = packet_sink_helper.Install(this->ue_nodes.Get(i));
        sink_apps.Start(Seconds(0));
    }
}

void RealNetworkScenario::setup_callbacks()
{
    // Create packet calculators for each UE, and set up callbacks to count the
    // number of bytes received over IPv4. Used by get_ue_rx_bytes() below
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        PacketSizeMinMaxAvgTotalCalculator *packet_calc = new PacketSizeMinMaxAvgTotalCalculator();
        this->ue_nodes.Get(i)->GetObject<Ipv4L3Protocol>()->TraceConnectWithoutContext("Rx",
            MakeBoundCallback(&RealNetworkScenario::callback_ipv4_packet_received, packet_calc));
        this->ue_packet_calcs.push_back(packet_calc);
    }

    // Connect callbacks to trigger whenever a UE is connected to a new eNodeB,
    // either because of initial network attachment or because of handovers
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
        MakeCallback(&RealNetworkScenario::callback_ue_spotted_at_enb, this));
    Config::Connect("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
        MakeCallback(&RealNetworkScenario::callback_ue_spotted_at_enb, this));

    // Connect callback for whenever an eNodeB receives "measurement reports".
    // These reports contain signal strength information of neighboring cells,
    // as seen by a UE. This is used by the eNodeB to determine handovers
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/RecvMeasurementReport",
        MakeCallback(&RealNetworkScenario::callback_measurement_report_received, this));
}

int RealNetworkScenario::get_ue_rx_bytes(int i)
{
    int rx_bytes = this->ue_packet_calcs[i]->getSum();
    this->ue_packet_calcs[i]->Reset();
    return rx_bytes;
}

void RealNetworkScenario::callback_ipv4_packet_received(
        PacketSizeMinMaxAvgTotalCalculator* packet_calc,
        Ptr<const Packet> packet, Ptr<Ipv4> ipv4, uint32_t iface)
{
    // Callback for each packet received at the IPv4 layer. Pass the packet
    // directly on to the PacketSizeMinMaxAvgTotalCalculator, which is used by
    // the periodic UE state reporting method via this->get_ue_rx_bytes()
    packet_calc->PacketUpdate("", packet);
}

void RealNetworkScenario::callback_ue_spotted_at_enb(
        std::string context, const uint64_t imsi,
        const uint16_t cell_id, const uint16_t rnti)
{
    // A given eNodeB (identified by cell ID) has become responsible for an UE
    // (identified by its IMSI), due to initial network attachment or handover
    std::cout << this->timestep() << " ms: UE seen at cell: "
        << "Cell " << (int)cell_id << " saw IMSI " << imsi << std::endl;
}

void RealNetworkScenario::callback_measurement_report_received(
        const uint64_t imsi, const uint16_t cell_id,
        const uint16_t rnti, const LteRrcSap::MeasurementReport report)
{
    // An eNodeB has received a measurement report of neighboring cell signal
    // strengths from an attached UE. Dump interesting information to stdout
    std::cout << this->timestep() << " ms: Measurement report: "
        << "Cell " << (int)cell_id
        << " got report from IMSI " << imsi
        << ": " << (int)cell_id
        << "/" << (int)report.measResults.rsrpResult
        << "/" << (int)report.measResults.rsrqResult;

    // There might be additional measurements to the one listed directly in the
    // data structure, hence we need to do some additional iteration
    for (auto iter = report.measResults.measResultListEutra.begin();
            iter != report.measResults.measResultListEutra.end(); iter++) {
        std::cout << " " << (int)iter->physCellId << "/"
            << (int)iter->rsrpResult << "/" << (int)iter->rsrqResult;
    }
    std::cout << std::endl;
}

//------------------------------------------------------------------------------
// Subclass for the mock network scenario
//------------------------------------------------------------------------------

class MockNetworkScenario : public BaseNetworkScenario
{
    public:
        void generate_map(std::string map_file) override;
        void enable_traces() override;

    protected:
        std::tuple<int, double> get_best_enb_at_coords(double x, double y);

        std::vector<int> enb_id_per_ue;
        std::vector<int> sinr_per_ue;
        std::vector<int> ue_count_per_enb;

        void create_lte_network() override;
        void apply_network_conf() override;
        void create_remote_server() override;
        void create_ue_applications() override;
        void setup_callbacks() override;
        int get_ue_rx_bytes(int i) override;

        void periodically_update_ue_assignments();
};

std::tuple<int, double> MockNetworkScenario::get_best_enb_at_coords(double x, double y)
{
    std::vector<double> enb_signals(this->enb_nodes.GetN());
    const double signal_coeff = 1.266338653300122e-07;
    const double background_noise = 1.4230e-13;

    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        double dx = x - this->enb_conf[i].x;
        double dy = y - this->enb_conf[i].y;
        double dz = this->enb_conf[i].z;
        double sq_dist = std::pow(dx, 2) + std::pow(dy, 2) + std::pow(dz, 2);

        double angle = std::atan2(dy, dx);
        angle -= this->enb_conf[i].direction * M_PI / 180.0;
        angle = std::atan2(std::sin(angle), std::cos(angle));
        double beamwidth = 70.0 * M_PI / 180.0;
        double loss = std::min(20.0, 12.0 * std::pow(angle / beamwidth, 2));

        double power = std::pow(10.0, (0.1 * this->enb_conf[i].tx_power - loss) / 10.0);
        enb_signals[i] = signal_coeff * power / sq_dist;
    }
    int best_enb = -1;
    double best_sinr = -1;
    for (uint32_t i = 0; i < this->enb_nodes.GetN(); i++) {
        double noise = background_noise;
        for (uint32_t j = 0; j < this->enb_nodes.GetN(); j++) {
            if (i != j) {
                noise += enb_signals[j];
            }
        }
        double sinr = enb_signals[i] / noise;

        if (sinr > best_sinr) {
            best_enb = i;
            best_sinr = sinr;
        }
    }
    return std::make_tuple(best_enb, best_sinr);
}

void MockNetworkScenario::generate_map(std::string map_file)
{
    int steps = 100;
    for (int col = 0; col < steps; col++) {
        for (int row = 0; row < steps; row++) {
            double x = -this->map_extent + (2 * this->map_extent) / (steps - 1) * col;
            double y = -this->map_extent + (2 * this->map_extent) / (steps - 1) * row;

            std::tuple<int, double> best_enb = this->get_best_enb_at_coords(x, y);
            std::cout << x << "\t" << y << "\t0\t" << std::get<1>(best_enb) << std::endl;
        }
    }
}

void MockNetworkScenario::enable_traces() {}
void MockNetworkScenario::create_lte_network() {}
void MockNetworkScenario::apply_network_conf() {}
void MockNetworkScenario::create_remote_server() {}
void MockNetworkScenario::create_ue_applications() {}

void MockNetworkScenario::setup_callbacks()
{
    this->periodically_update_ue_assignments();
}

int MockNetworkScenario::get_ue_rx_bytes(int i)
{
    const int max_enb_bytes = 66603;
    int current_enb = this->enb_id_per_ue[i];
    return max_enb_bytes / this->ue_count_per_enb[current_enb];
}

void MockNetworkScenario::periodically_update_ue_assignments()
{
    this->enb_id_per_ue.resize(this->ue_nodes.GetN(), -1);
    this->sinr_per_ue.resize(this->ue_nodes.GetN());
    this->ue_count_per_enb.assign(this->enb_nodes.GetN(), 0);

    // Determine UE-to-eNB assignments, print out the base station assignments
    // and calculate number of users per base station (used by get_ue_rx_bytes)
    for (uint32_t i = 0; i < this->ue_nodes.GetN(); i++) {
        int best_enb; double best_sinr;
        Vector position = this->ue_nodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
        std::tie(best_enb, best_sinr) = this->get_best_enb_at_coords(position.x, position.y);

        if (this->enb_id_per_ue[i] != best_enb) {
            std::cout << this->timestep() << " ms: UE seen at cell: "
                << "Cell " << (best_enb + 1) << " saw IMSI " << (i + 1) << std::endl;
        }

        this->enb_id_per_ue[i] = best_enb;
        this->sinr_per_ue[i] = best_sinr;
        this->ue_count_per_enb[best_enb]++;
    }

    // Reschedule again after 480 ms, matching the update interval of the real
    // handover algorithm's A4 event in a2-a4-rsrq-handover-algorithm.cc:127
    Simulator::Schedule(MilliSeconds(480),
        &MockNetworkScenario::periodically_update_ue_assignments, this);
}

//------------------------------------------------------------------------------
// Simulator entry point, command line parsing etc.
//------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    // Parse command line arguments
    bool mock_scenario = false;
    std::string map_file = "";
    bool enable_traces = false;
    int warmup_duration = 4000;
    int interaction_interval = 100;
    double enb_inter_distance = 1000.0;
    int left_enb_power = 300;
    int right_enb_power = 300;
    int top_enb_power = 300;
    int cluster_count_max = 3;
    int ue_count_min = 3;
    int ue_count_max = 5;
    std::string seeds_tmp = "1";
    std::vector<std::uint64_t> seeds;
    int total_ue_count = 12;
    int seed_pause = 30000;
    double speed = 1.4 * 10;
    int min_power = 200;
    int max_power = 400;

    CommandLine cmd;
    cmd.AddValue("mock", "Use the mock version of the network scenario", mock_scenario);
    cmd.AddValue("map", "Render the Radio Environment Map to this file and quit", map_file);
    cmd.AddValue("traces", "Enable traces from the LTE module to generate *Stats.txt files", enable_traces);
    cmd.AddValue("warmup", "Warmup duration before starting agent interaction (in ms)", warmup_duration);
    cmd.AddValue("interval", "Agent interaction interval (in ms)", interaction_interval);
    cmd.AddValue("distance", "Distance between adjacent eNodeBs (in m)", enb_inter_distance);
    cmd.AddValue("left", "Transmission power for the left eNodeB (in dBm)", left_enb_power);
    cmd.AddValue("right", "Transmission power for the right eNodeB (in dBm)", right_enb_power);
    cmd.AddValue("top", "Transmission power for the top eNodeB (in dBm)", top_enb_power);
    cmd.AddValue("cluster", "Maximum number of UE clusters", cluster_count_max);
    cmd.AddValue("min", "Minimum size of a given UE cluster", ue_count_min);
    cmd.AddValue("max", "Maximum size of a given UE cluster", ue_count_max);
    cmd.AddValue("seeds", "Comma-separated list of RNG seeds for defining motion", seeds_tmp);
    cmd.AddValue("ue_count", "Total number of UEs", total_ue_count);
    cmd.AddValue("pause", "Pause duration between seeds (in ms)", seed_pause);
    cmd.AddValue("speed", "UE movement speed (in m/s)", speed);
    cmd.AddValue("min_power", "Minimum allowed power setting (same units as left/right/top)", min_power);
    cmd.AddValue("max_power", "Maximum allowed power setting (same units as left/right/top)", max_power);
    cmd.Parse(argc, argv);

    std::vector<std::string> seeds_tmp_2 = forlorn::split(seeds_tmp, ',');
    for (auto i = 0; i < seeds_tmp_2.size(); ++i) { seeds.push_back(std::stoul(seeds_tmp_2[i])); }

    // Instantiate either the real or the mock version of the scenario
    BaseNetworkScenario *scenario;
    if (mock_scenario) {
        std::cout << "Initializing mock simulator" << std::endl;
        scenario = new MockNetworkScenario();
    } else {
        std::cout << "Initializing real simulator" << std::endl;
        scenario = new RealNetworkScenario();
    }

    // Initialize the scenario (constructing network nodes, applications and callbacks)
    scenario->initialize(
        warmup_duration, interaction_interval, enb_inter_distance,
        left_enb_power, right_enb_power, top_enb_power,
        ue_count_min, ue_count_max, total_ue_count,
        seed_pause, seeds, speed, min_power, max_power);

    // If requested on command line, render a Radio Environment Map and quit
    if (map_file.length() > 0) {
        scenario->generate_map(map_file);
        return 0;
    }

    // If requested, enable trace outputs from the LTE module. This will
    // generate a bunch of Dl*Stats.txt/Ul*Stats.txt files in the working
    // directory, containing textual representations of most of the internal
    // trace sources supported by the LTE stack (such as within the PDCP/RLC
    // layers, MAC scheduling, UL/DL interference etc.)
    if (enable_traces) {
        scenario->enable_traces();
    }

    // Run the scenario and quit
    try {
        scenario->run();
    } catch (std::invalid_argument) {
        std::cout << "Simulator exiting" << std::endl;
    }
    return 0;
}
