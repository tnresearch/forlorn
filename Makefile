# FORLORN - Comparing Offline Methods and RL for RAN Parameter Optimization
# Copyright (c) 2022 Telenor ASA
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# -----
#
# This code accompanies the following paper: Vegard Edvardsen, Gard Spreemann,
# and Jeriek Van den Abeele. 2022. FORLORN: A Framework for Comparing Offline
# Methods and Reinforcement Learning for Optimization of RAN Parameters.
# Submitted to the 25th ACM International Conference on Modeling, Analysis and
# Simulation of Wireless and Mobile Systems (MSWiM '22).
#
# -----
#
# Makefile - Building the simulator and linking to precompiled ns-3 library



# Which ns-3 version to use. If nothing else is specified, our implementation
# assumes ns3.31, which matches libns3-dev's version in Debian 11 (bullseye)

NS3_PREFIX ?= ns3.31

# Our required ns-3 modules

NS3_MODULES += applications
NS3_MODULES += core
NS3_MODULES += csma
NS3_MODULES += internet
NS3_MODULES += internet-apps
NS3_MODULES += lte
NS3_MODULES += mobility
NS3_MODULES += network
NS3_MODULES += point-to-point
NS3_MODULES += propagation
NS3_MODULES += spectrum
NS3_MODULES += stats

# Objects to compile

TARGET = simulator
OBJS += simulator.o
OBJS += mobility.o

# Compilation flags

NS3_BUILD_PROFILE ?= RELEASE
ifeq ($(NS3_BUILD_PROFILE), DEBUG)
    CXXFLAGS += -DNS3_ASSERT_ENABLE -DNS3_LOG_ENABLE
endif

CXXFLAGS += -I/usr/include/$(NS3_PREFIX) -DNS3_BUILD_PROFILE_${NS3_BUILD_PROFILE}
LDLIBS += $(NS3_MODULES:%=-l$(NS3_PREFIX)-%)

# Rules

build: $(TARGET)
clean:
	$(RM) $(TARGET) $(OBJS)
.PHONY: build clean

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDLIBS)
