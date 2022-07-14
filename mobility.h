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
// mobility.h - Headers for our custom mobility model

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <ns3/double.h>
#include <ns3/event-id.h>
#include <ns3/mobility-model.h>
#include <ns3/nstime.h>
#include <ns3/simulator.h>
#include <ns3/vector.h>

namespace forlorn
{
  inline std::vector<std::string> split(const std::string & s, char delim)
  {
    std::vector<std::string> ret;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
      ret.push_back(item);
    }
    return ret;
  }

  inline void scale(double c, ns3::Vector3D & v)
  {
    v.x *= c;
    v.y *= c;
    v.z *= c;
  }

  inline void normalize(ns3::Vector3D & v)
  {
    scale(1.0/v.GetLength(), v);
  }

  class DirectMobilityModel : public ns3::MobilityModel
  {
  public:
    static ns3::TypeId GetTypeId(void);
    DirectMobilityModel();
    virtual ~DirectMobilityModel();

    inline bool has_arrived() const { return ns3::Simulator::Now() >= arrival_time; }
    inline ns3::Vector get_from() const { return from; }
    inline void set_from(ns3::Vector p) { DoSetPosition(p); }
    inline ns3::Vector get_to() const { return to; }
    inline void set_to(ns3::Vector p) { ns3::Vector tmp = DoGetPosition(); to = p; DoSetPosition(tmp); }
    inline ns3::Time get_arrival_time() const { return arrival_time; }

    virtual ns3::Vector DoGetPosition() const;
    virtual ns3::Vector DoGetVelocity() const;
    virtual void DoSetPosition(const ns3::Vector & position);

  protected:
    double speed;

  private:
    ns3::Vector from;
    ns3::Vector to;
    ns3::Time start_time;
    ns3::Time arrival_time;
    ns3::EventId arrival_event;

    void HandleArrival();
  };
}
