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
// mobility.cc - Implementation of our custom mobility model

#include <iostream>
#include <limits>
#include <string>

#include <ns3/double.h>
#include <ns3/log.h>
#include <ns3/nstime.h>
#include <ns3/simulator.h>

#include "mobility.h"

namespace forlorn
{
  NS_LOG_COMPONENT_DEFINE("DirectMobilityModel");
  NS_OBJECT_ENSURE_REGISTERED(DirectMobilityModel);

  ns3::TypeId DirectMobilityModel::GetTypeId()
  {
    static ns3::TypeId tid = ns3::TypeId("forlorn::DirectMobilityModel")
      .SetParent<ns3::MobilityModel>()
      .SetGroupName("Mobility")
      .AddConstructor<DirectMobilityModel>()
      .AddAttribute ("From", "Starting point.",
                     ns3::VectorValue(),
                     ns3::MakeVectorAccessor(&DirectMobilityModel::set_from, &DirectMobilityModel::get_from),
                     ns3::MakeVectorChecker())
      .AddAttribute ("To", "Ending point.",
                     ns3::VectorValue(),
                     ns3::MakeVectorAccessor(&DirectMobilityModel::set_to, &DirectMobilityModel::get_to),
                     ns3::MakeVectorChecker())
      .AddAttribute ("Speed", "Walker speed.",
                     ns3::DoubleValue(1.4), // m/s
                     ns3::MakeDoubleAccessor(&DirectMobilityModel::speed),
                     ns3::MakeDoubleChecker<double>(0.0, std::numeric_limits<double>::infinity()));
      return tid;
  }

  DirectMobilityModel::DirectMobilityModel()
  : from(0,0,0), to(0,0,0), speed(0)
  {
  }

  DirectMobilityModel::~DirectMobilityModel()
  {
  }

  ns3::Vector DirectMobilityModel::DoGetPosition() const
  {
    ns3::Time t = ns3::Simulator::Now();
    if (t >= arrival_time)
    {
      return to;
    }
    else if (t < start_time)
    {
      NS_ABORT_MSG("Quit living in the past, man!");
    }
    else
    {
      ns3::Vector travelled = to - from;
      scale((speed*(t - start_time).GetSeconds())/travelled.GetLength(), travelled);
      return from + travelled;
    }
  }

  ns3::Vector DirectMobilityModel::DoGetVelocity() const
  {
    if (has_arrived())
      return ns3::Vector(0,0,0);
    else
    {
      ns3::Vector velocity = to - from;
      scale(speed/velocity.GetLength(), velocity);
      return velocity;
    }
  }

  void DirectMobilityModel::HandleArrival()
  {
    //std::cout << "ARRIVAL" << std::endl;
  }

  void DirectMobilityModel::DoSetPosition(const ns3::Vector & position)
  {
    //std::cout << "DoSetPosition" << std::endl;
    from = position;
    start_time = ns3::Simulator::Now();
    double distance = ns3::CalculateDistance(from, to);
    //std::cout << "with from=[" << from.x << "," << from.y << "]   and   to=[" << to.x << "," << to.y << "]" << std::endl;
    //std::cout << "distance=" << distance << std::endl;
    if (distance > 0)
    {
      ns3::Time arrival_delay(speed > 0 ? (std::to_string(distance/speed) + std::string("s")) : std::string("1337y"));
      arrival_event.Cancel();
      //std::cout << "Scheduling arrival in " << arrival_delay << std::endl;
      arrival_event = ns3::Simulator::Schedule(arrival_delay, &DirectMobilityModel::HandleArrival, this);
      arrival_time = start_time + arrival_delay;
    }
    else
    {
      arrival_time = start_time;
    }
    NotifyCourseChange();
  }
}
